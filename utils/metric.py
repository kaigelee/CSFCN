"""Evaluation Metrics for Semantic Segmentation"""
import torch
import numpy as np

__all__ = ['SegmentationMetric_MY', 'hist_info', 'compute_score','print_iou']




class SegmentationMetric_MY(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass=19):
        super(SegmentationMetric_MY, self).__init__()
        self.nclass = nclass
        self.reset()
        if nclass == 19:
            self.name = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        elif nclass==11:
            self.name = ['Sky','Building','Pole','Road','Pavement',
                         'Tree','SignSymbol','Fence','Car','Pedestrian',
                         'Bicyclist']



    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            pred = torch.argmax(pred, 1)

            pred = pred.cpu().data.numpy()

            hist,labeled, correct = hist_info(pred,label.cpu().data.numpy(),num_cls=self.nclass)


            self.hist+= hist
            self.total_label += labeled
            self.total_correct += correct


        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get_show(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        iu,mean_IU, mean_pixel_acc = compute_score(self.hist, correct=self.total_correct, labeled=self.total_label)
        #
        # result_line = print_iou(iu, mean_IU,mean_pixel_acc,
        #                         self.name)


        return mean_pixel_acc,mean_IU #,result_line

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.hist = np.zeros((self.nclass,self.nclass))
        self.total_correct = 0
        self.total_label = 0

def hist_info(pred, label, num_cls=19):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct

def compute_score(hist, correct=0, labeled=1):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    # mean_IU_no_back = np.nanmean(iu[1:])
    # freq = hist.sum(1) / hist.sum()
    # freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu,mean_IU, mean_pixel_acc
def print_iou(iu, mean_IU,mean_pixel_acc, class_names=None):
    n = iu.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i + 1)
        else:
            cls = '%d %s' % (i + 1, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iu[i] * 100))

    lines.append(
            '----------------------------\n%-8s\t%.3f%%\t%-8s\t%.3f%%' % (
                'mean_IU', mean_IU * 100, 'mean_pixel_ACC',
                mean_pixel_acc * 100))
    line = "\n".join(lines)
    # print(line)
    return line
