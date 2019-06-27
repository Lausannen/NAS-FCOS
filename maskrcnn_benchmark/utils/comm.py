"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time

import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def compute_params(model):
    """Compute number of parameters"""
    n_total_params = 0
    n_aux_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        if ('aux' in name):
            n_aux_params += n_elem
        n_total_params += n_elem
    return n_total_params, n_total_params - n_aux_params


def encode(encoded_data, data):
    """Encode data which will be passed between different process
    
    Arguments:
        encoded_data (list): encoded_data has max_buffer with 2048 B
            encoded_data[0] records useful space length
        data (tensor): data which will firstly be encoded and then be passed
    """
    # gets a byte representation for the data
    encoded_bytes = pickle.dumps(data)
    # convert this byte string into a byte tensor
    storage = torch.ByteStorage.from_buffer(encoded_bytes)
    tensor = torch.ByteTensor(storage).to("cuda")
    # encoding: first byte is the size and then rest is the data
    s = tensor.numel()
    assert s <= 2047, "Can't encode data greater than 2047 bytes"
    # put the encoded data in encoded_data
    encoded_data[0] = s
    encoded_data[1 : (s + 1)] = tensor


def decode(encoded_data):
    """Decode data ,usually used in process or gpu which receive data coming from
        main process
    
    Arguments:
        encoded_data (list): encoded data coming from main process
    
    Returns:
        data (tensor): decoded data
    """
    size = encoded_data[0]
    encoded_tensor = encoded_data[1 : (size + 1)].to("cpu")
    return pickle.loads(bytearray(encoded_tensor.tolist()))


def config2action(config, decoder_version):
    """Transfer configure to action
    
    Arguments:
        config (list): a list which has list elements
    
    Returns:
        action (list): all of its element are numeral type
    """
    if decoder_version == 2:
        _, dec_arch = config
        ctx = dec_arch
        action = []
        for cell in ctx:
            lead = cell[0]
            action += lead
        return action
    elif decoder_version == 3:
        _, config = config
        action = []
        for _, elem in enumerate(config):
            action += elem[:2]
            cell = elem[2]
            for c_elem in cell:
                action += c_elem
        return action
    else:
        raise ValueError("Do not support transfer Decoder Version {}".format(decoder_version))


def action2config(action, decoder_version, enc_end=0, dec_block=3, ctx_block=4):
    """Transfer action to configure
    Note: parameters should be modified if cell and layer info are changed

    Arguments:
        action (list): a list contain configure info
        enc_end (int): index of encode_config info end
        dec_block (int): numbers of cell in decoder.fpn structure
        ctx_block (int): numbers of layer in single cell
    
    Returns:
        config (list): all of its element are numeral type
    """
    if decoder_version == 2:
        ctx = []
        branch_size = len(action) // dec_block
        assert branch_size == 5, "Branch size must be 5, got {}".format(branch_size)
        for i in range(dec_block):
            branch = []
            branch.append(action[(i * branch_size) : ((i + 1) * branch_size)])
            ctx.append(branch)                        
        return None, ctx

    elif decoder_version == 3:
        config = []
        for l in range(dec_block):
            conns = []
            ops = []
            cells = []
            for i in range(2):
                # global connections
                conns.append(action[4 * (ctx_block + 1) * l + i])
            for i in range(2):
                # first ops
                ops.append(action[4 * (ctx_block + 1) * l + 2 + i])
            for ll in range(ctx_block):
                cell = []
                for j in range(4):
                    # cells
                    cell.append(action[4 * (ctx_block + 1) * l + 4 * (ll + 1) + j])
                cells.append(cell)
            config.append(conns + [[ops] + cells])
        return None, config

    else:
        raise ValueError("Do not support transfer Decoder Version {}".format(decoder_version))
