import numpy as np
import os
import shutil
import torch
import subprocess
import torch.distributed as dist
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from models import MyCLIP
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def get_gpu_info():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

def convert_image_to_rgb(image):
    return image.convert("RGB")

def transform(n_px, train=True):
    if train:
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            RandomResizedCrop(n_px, scale=(0.8, 1.0)),
            convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def recall_at_k(score_matrix, prefix = 't2i_', file_names=None):
    '''
    Calculating the final R@K scores for image-text retrieval or text-image retrieval.
    The row elements are taken as queries.
    Input:
        score_matrix [torch.tensor]: a matrix 
    Return:
        R@1, R@5, R@10 and R@mean for both image-text retrieval or text-image retrieval.
    '''

    # image indexes
    score_matrix.type(torch.float32)
    m_shape = score_matrix.shape
    # top 10 image indexes
    _, rank_txt_idx = score_matrix.topk(10, dim=1)
    print('rank_txt_idx.shape', rank_txt_idx.shape)
    retrieved_files = []
    if not file_names == None:
        for row in range(rank_txt_idx.shape[0]):
            files_for_the_text = []
            for column in range(rank_txt_idx.shape[1]):
                files_for_the_text.append(file_names[rank_txt_idx[row, column]])
            retrieved_files.append(files_for_the_text)

    # ground truth of image indexes, each row gets extended
    gt_img_j = torch.LongTensor([i for i in range(m_shape[0])]).unsqueeze(1).expand_as(rank_txt_idx)
    # non-zero element indexes
    # nonzero() Return the indices of the elements that are non-zero.
    # rank.shape = (rows, 2), (:, 0) are the values, (:, 1) are the indices 
    rank = (rank_txt_idx == gt_img_j).nonzero()[:, 1]
    if rank.numel():
        r1 = (rank < 1).sum().item() / m_shape[0]
        r5 = (rank < 5).sum().item() / m_shape[0]
        r10 = (rank < 10).sum().item() / m_shape[0]
        r_mean = (r1 + r5 + r10) / 3
    else:
        r1, r5, r10, r_mean = 0, 0, 0, 0
    eval_log = {prefix+'r1': r1,
                prefix+'r5': r5,
                prefix+'r10': r10,
                prefix+'r_mean': r_mean
                }
    return eval_log, retrieved_files

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
      or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    elif isinstance(model, MyCLIP):
        return model.model
    else:
        return model


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = f'{output_dir}/checkpoint.pt'
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters + 1) # When accumulation is True, we need an extra more LR
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep + 1
    return schedule


def visualize_retrieved_memes(retrieved_files, query_texts, target_images, output_file="output.html"):
    '''
    Given the file names of retrieved images and the query texts, visualize the results.
    
    Parameters:
      retrieved_files: list of lists; each element is a list of 5 image filename strings.
      query_texts: list of strings; each element is the query text for a row.
      target_images: list of strings; each element is a target image filename string.
      output_file: name of the output HTML file (default "output.html")
    
    The output HTML file will contain a table with the following columns:
       - Query text
       - Target image
       - Retrieved image 1
       - Retrieved image 2
       - Retrieved image 3
       - Retrieved image 4
       - Retrieved image 5
       
    For the target and retrieved images, the base file name is displayed below the image.
    '''
    # Start the HTML string
    html_str = """
    <html>
      <head>
        <meta charset="UTF-8">
        <title>Visualization of Retrieved Memes</title>
        <style>
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
          img { max-width: 150px; height: auto; display: block; margin: 0 auto; }
          .filename { font-size: 0.9em; color: #555; }
        </style>
      </head>
      <body>
        <h2>Visualization of Retrieved Memes</h2>
        <table>
          <tr>
            <th>Query text</th>
            <th>Target image</th>
            <th>Retrieved image 1</th>
            <th>Retrieved image 2</th>
            <th>Retrieved image 3</th>
            <th>Retrieved image 4</th>
            <th>Retrieved image 5</th>
          </tr>
    """

    # Loop over each row of data (assuming lengths match)
    for query, target, retrieved in zip(query_texts, target_images, retrieved_files):
        # print(query, target, retrieved)
        html_str += "<tr>"
        # Query text cell
        html_str += f"<td>{query}</td>"
        # Target image cell with image and base filename below
        target_base = os.path.basename(target)
        html_str += f"<td><img src='{target}' alt='Target Image'><div class='filename'>{target_base}</div></td>"
        # For each of the 5 retrieved images
        for img in retrieved:
            img_base = os.path.basename(img)
            html_str += f"<td><img src='{img}' alt='Retrieved Image'><div class='filename'>{img_base}</div></td>"
        html_str += "</tr>"

    # Close table and HTML
    html_str += """
        </table>
      </body>
    </html>
    """

    # Write the HTML string to a file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"HTML file created: {output_file}")
