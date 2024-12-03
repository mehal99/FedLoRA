import torch

sig = torch.nn.Sigmoid()

def test_batch_cls(model, x, y, multilabel=False): # classification
    outputs = model(x, labels=y)
    logits = outputs.logits.detach()
    loss = outputs.loss # huggingface loss is already averaged
    preds = logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    # stats = {
    #     'tp': correct,
    #     'fp': len(y) - correct,
    #     'fn': len(y) - correct,
    #     'count': x.shape[0],
    #     'loss': loss.item()*x.shape[0],
    # }
    return loss, correct

def eval_loop(model, loader):        
    model.eval()
    count, acc, total_loss = 0, 0, 0
    for x,y in loader:
        x, y = x.to("cuda"), y.to("cuda")
        count += len(y)
        with torch.no_grad():
            loss, correct = test_batch_cls(model, x, y)
            total_loss += loss.item()
            acc += correct
    acc /= count
    total_loss /= count
    return acc, total_loss

# def get_metric(stats, metric):
#         if stats['tp'] == 0:
#             return 0
#         elif metric == 'accu':
#             return stats['tp'] / (stats['tp'] + stats['fp'])
#         elif metric == 'recall':
#             return stats['tp'] / (stats['tp'] + stats['fn'])
#         elif metric == 'f1':
#             return 2*stats['tp'] / (2*stats['tp'] + stats['fp'] + stats['fn'])
    
# def log_stats(writer, prefix, stats, step):
#     with writer.as_default():
#         tf.summary.scalar(f"{prefix}/accuracy", get_metric(stats, 'accu'), step=step)
#         tf.summary.scalar(f"{prefix}/recall", get_metric(stats, 'recall'), step=step)
#         tf.summary.scalar(f"{prefix}/f1", get_metric(stats, 'f1'), step=step)
#         for k,v in stats.items():
#             if k not in ['tp', 'fp', 'tn', 'fn']:
#                 tf.summary.scalar(f"{prefix}/{k}", v, step=step)