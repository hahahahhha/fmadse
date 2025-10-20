import torch
import torch.nn as nn
from typing import Optional, List
import time


def _generate_fp_values(exponent_bits: int, mantissa_bits: int) -> List[float]:
    """Generate sorted representable values for an FP format excluding inf/nan."""
    bias = (1 << (exponent_bits - 1)) - 1
    scale = 1 << mantissa_bits
    positives = [0.0]
    # Subnormal positives (exponent==0, mantissa!=0)
    subnormal_factor = 2 ** (1 - bias)
    for mantissa in range(1, scale):
        positives.append(subnormal_factor * mantissa / scale)
    # Normalized positives (1 <= exponent <= max-1)
    for exponent in range(1, (1 << exponent_bits) - 1):
        exp_factor = 2 ** (exponent - bias)
        for mantissa in range(scale):
            positives.append(exp_factor * (1 + mantissa / scale))
    negatives = [-value for value in reversed(positives) if value != 0.0]
    return negatives + positives

def _generate_int_values(bits: int) -> List[float]:
    """Generate symmetric INT levels for given bit-width (signed, with zero)."""
    qmax = (1 << (bits - 1)) - 1
    return [float(i) for i in range(-qmax, qmax + 1)]


def _generate_rounded_fp_codebook(bits: int, mantissa_bits: int) -> List[float]:
    """
    Generate symmetric FP-like codebook with integer levels using rounding.
    - Start from exponent e=0 upward; for each e, enumerate mantissas m in [0, 2^mantissa_bits).
    - Level = round((1 + m / 2^mantissa_bits) * 2^e).
    - Keep unique positive integers in ascending order until we have (2^(bits-1) - 1) positives.
    - Return symmetric list: negatives (descending), 0, positives (ascending).
    """
    target_pos = (1 << (bits - 1)) - 1
    pos: List[float] = []
    seen = set()
    scale = 1 << mantissa_bits
    e = 0
    while len(pos) < target_pos:
        base = 1 << e
        # enumerate mantissas in increasing order to keep ascending sequence
        for m in range(scale):
            val = int(round(base * (1.0 + m / scale)))
            if val <= 0:
                val = 1
            if val not in seen:
                seen.add(val)
                pos.append(float(val))
                if len(pos) >= target_pos:
                    break
        e += 1
    neg = [-v for v in reversed(pos)]
    return neg + [0.0] + pos

#################################  3-bit Datatypes  #################################
FP3 = _generate_fp_values(2, 0)

#################################  4-bit Datatypes  #################################
INT4 = _generate_int_values(4)
FP4_E2M1 = _generate_fp_values(2, 1)

#################################  6-bit Datatypes  #################################
FP6_E2M3 = _generate_fp_values(2, 3)
FP6_E3M2 = _generate_fp_values(3, 2)

#################################  8-bit Datatypes  #################################
INT8 = _generate_int_values(8)
FP8_E5M2 = _generate_fp_values(5, 2)
FP8_E4M3 = _generate_fp_values(4, 3)

DATATYPE_MAPPING_3_BIT = {
     'fp3': FP3, 
}
DATATYPE_MAPPING_3_BIT_MX = {
     'mx_fp3': FP3
}

DATATYPE_MAPPING_4_BIT = {
    'int4': INT4, 'fp4': FP4_E2M1,
}
DATATYPE_MAPPING_4_BIT_MX = {
     'mx_fp4': FP4_E2M1 
}

DATATYPE_MAPPING_6_BIT = {
    'fp6_e2m3': FP6_E2M3, 'fp6_e3m2': FP6_E3M2
}

DATATYPE_MAPPING_6_BIT_MX = {
    'mx_fp6_e2m3': FP6_E2M3, 'mx_fp6_e3m2': FP6_E3M2
}

DATATYPE_MAPPING_8_BIT = {
    'int8': INT8,
    'fp8_e5m2': FP8_E5M2,
    'fp8_e4m3': FP8_E4M3,
}

DATATYPE_MAPPING_8_BIT_MX = {
    'mx_fp8_e5m2': FP8_E5M2,
    'mx_fp8_e4m3': FP8_E4M3,
}

WQ_BIT_MAPPING_DATATYPE={
    3:['fp3'],
    4:['int4','fp4'],
    6:['fp6_e2m3','fp6e3m2'],
    8:['fp8_e4m3','fp8e5m2','int8']
}

WQ_BIT_MAPPING_DATATYPE_MX={
    3:['mx_fp3'],
    4:['mx_fp4'],
    6:['mx_fp6_e2m3','mx_fp6e3m2'],
    8:['mx_fp8_e4m3','mx_fp8e5m2']
}

@torch.no_grad()
def quant_int(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Symmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)

    w_fp16_new = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_int_asym(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Asymmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**wq_bits - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    zeropoint = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=qmin, max=qmax)

    w_fp16_new = (q_tensor - zeropoint) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_mx(w_fp16, wq_bits:int=4, datatype: str="", group_size: int=32):
    """
        MX quantization.
        Reference: https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py
    """ 
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_MX
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_MX
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT_MX
    elif wq_bits == 8:
        DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT_MX
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit, 6-bit, 8-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size() # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
    
    shared_exp, _ = torch.max(w_fp16_new.abs(), dim=-1, keepdim=True)
    shared_exp = torch.floor(torch.log2(shared_exp))
    w_fp16_new = w_fp16_new / (2**shared_exp)
    qmax = max([abs(x) for x in allow_value])
    scale = 1 / (qmax / 2)
    x = w_fp16_new / scale

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale * (2**shared_exp)
    return w_fp16_new.reshape(K, C).to(torch.float16)


@torch.no_grad()
def quant_datatype(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None,if_mx=False):
    if if_mx:
        if wq_bits == 3:
            DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_MX
        elif wq_bits == 4:
            DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_MX
        elif wq_bits == 6:
            DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT_MX
        elif wq_bits == 8:
            DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT_MX
        else:
            raise ValueError(f"Currently only support 3-, 4-, 6-, and 8-bit mx quantization, not {wq_bits}-bit")
    else:
        if wq_bits == 3:
            DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
        elif wq_bits == 4:
            DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
        elif wq_bits == 6:
            DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
        elif wq_bits == 8:
            DATATYPE_MAPPING = DATATYPE_MAPPING_8_BIT
        else:
            raise ValueError(f"Currently only support 3-, 4-,  6-, and 8-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    if if_mx:
        if group_size is None or group_size<=0:
            w_fp16_new=w_fp16.to(torch.float32)
        else:
            K, C = w_fp16.size() # output channel, input channel
            NUM_GROUP = C // group_size
            w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
        
        shared_exp, _ = torch.max(w_fp16_new.abs(), dim=-1, keepdim=True)
        shared_exp = torch.floor(torch.log2(shared_exp))
        w_fp16_new = w_fp16_new / (2**shared_exp)
        qmax = max([abs(x) for x in allow_value])
        scale = 1 / (qmax / 2)
        x = w_fp16_new / scale

        q_tensor = torch.zeros_like(x)
        for i in range(len(allow_value)):
            data = allow_value[i]
            if i == 0:
                q_tensor += torch.where(x <= mid_value[i], data, 0)
            elif i == len(allow_value) - 1:
                q_tensor += torch.where(x > mid_value[i - 1], data, 0)
            else:
                q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

        w_fp16_new = q_tensor * scale * (2**shared_exp)
        if (group_size is None) or (group_size <= 0):
            return w_fp16_new.to(torch.float16)
        else:
            return w_fp16_new.reshape(K, C).to(torch.float16)
    else:  
        if (group_size is None) or (group_size <= 0):
            w_fp16_new = w_fp16.to(torch.float16)
        else:
            K, C = w_fp16.size() # output channel, input channel
            NUM_GROUP = C // group_size
            w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)

        rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
        qmax = max([abs(x) for x in allow_value])
        scale_fp = rmax / qmax
        scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
        x = w_fp16_new / scale_fp

        q_tensor = torch.zeros_like(x)
        for i in range(len(allow_value)):
            data = allow_value[i]
            if i == 0:
                q_tensor += torch.where(x <= mid_value[i], data, 0)
            elif i == len(allow_value) - 1:
                q_tensor += torch.where(x > mid_value[i - 1], data, 0)
            else:
                q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

        w_fp16_new = q_tensor * scale_fp 

        if (group_size is None) or (group_size <= 0):
            return w_fp16_new
        else:
            return w_fp16_new.reshape(K, C)


@torch.no_grad()
def search_datatype(w_fp16, wq_bits:int=4, datatype_support: List[str]=['fp3','fp4','fp6_e2m3','fp6_e3m2','fp8_e4m3','fp8_e5m2','int4','int8'],if_mx_support:bool=False,
                     group_size: Optional[int]=None):
    
    if wq_bits in WQ_BIT_MAPPING_DATATYPE:
        datatype_list=[datatype for datatype in WQ_BIT_MAPPING_DATATYPE[wq_bits] if datatype in datatype_support]
    else:
        raise ValueError(f"Currently only support {', '.join([str(k)+'-bit' for k in WQ_BIT_MAPPING_DATATYPE])} mixed quantization, not {wq_bits}-bit")

    K, C = w_fp16.size() # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)
    
    error = torch.full([K, NUM_GROUP], 1e3, dtype=w_fp16.dtype, device=w_fp16.device)
    for datatype in datatype_list:
        if if_mx_support and ('mx_'+datatype in WQ_BIT_MAPPING_DATATYPE_MX[wq_bits]):
            w_fp16_tmp = quant_datatype(w_fp16, wq_bits=wq_bits, datatype='mx_'+datatype, group_size=None, if_mx=True)
        else:
            w_fp16_tmp = quant_datatype(w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None, if_mx=False)
        quant_error = (w_fp16_tmp - w_fp16).pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_tmp[update_mask]

        del w_fp16_tmp, quant_error, update_mask
    
    return q_tensor.reshape(K, C)


def quant_model(model, wq_bits: Optional[int]=None, wq_datatype: Optional[str]=None, wq_groupsize: Optional[int]=None):
    if (wq_datatype is None) or (wq_datatype in ["fp16", "fp32"]):
        print("Not applying quantization")
        time.sleep(2)
    elif (wq_datatype.startswith("int")) and ("asym" in wq_datatype):
        print(f"Applying asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int_asym(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif (wq_datatype.startswith("int")) and ("asym" not in wq_datatype):
        print(f"Applying symmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif ("mx" in wq_datatype):
        '''
            We use hard-coded group size 32 based on the Open Compute Standard
            https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
        '''
        print(f"Applying MX quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: 32")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_mx(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=32)
    else:
        print(f"Applying mixed datatype quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                print(f'Quantizing layer: {n}')
                m.weight.data = search_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)
