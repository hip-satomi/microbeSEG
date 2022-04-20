import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(loss_function, label_type):
    """ Get loss function(s) for the training process.

    :param loss_function: Loss function to use.
        :type loss_function: str
    :param label_type: Label type of the training data / predictions, e.g., 'boundary'
        :type label_type: str
    :return: Loss function / dict of loss functions.
    """

    if label_type == 'boundary':
        if loss_function == 'ce_dice':
            criterion = ce_dice
        elif loss_function == 'ce':
            criterion = nn.CrossEntropyLoss()
        else:
            raise Exception('Loss unknown')
    elif label_type == 'distance':
        if loss_function == 'l1':
            border_criterion = nn.L1Loss()
            cell_criterion = nn.L1Loss()
        elif loss_function == 'l2':
            border_criterion = nn.MSELoss()
            cell_criterion = nn.MSELoss()
        elif loss_function == 'smooth_l1':
            border_criterion = nn.SmoothL1Loss()
            cell_criterion = nn.SmoothL1Loss()
        else:
            raise Exception('Loss unknown')
        criterion = {'border': border_criterion, 'cell': cell_criterion}

    return criterion


def dice_loss(y_pred, y_true, use_sigmoid=True):
    """Dice loss: harmonic mean of precision and recall (FPs and FNs are weighted equally). Only for 1 output channel.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, channels, height, width].
        :type y_true:
    :param use_sigmoid: Apply sigmoid activation function to the prediction y_pred.
        :type use_sigmoid: bool
    :return:
    """

    # Avoid division by zero
    smooth = 1.

    # Flatten ground truth
    gt = y_true.contiguous().view(-1)

    if use_sigmoid:  # Apply sigmoid activation to prediction and flatten prediction
        pred = torch.sigmoid(y_pred)
        pred = pred.contiguous().view(-1)
    else:
        pred = y_pred.contiguous().view(-1)

    # Calculate Dice loss
    pred_gt = torch.sum(gt * pred)
    loss = 1 - (2. * pred_gt + smooth) / (torch.sum(gt ** 2) + torch.sum(pred ** 2) + smooth)

    return loss


def ce_dice(y_pred, y_true, num_classes=3):
    """Sum of crossentropy loss and channel-wise Dice loss.

    :param y_pred: Prediction [batch size, channels, height, width].
        :type y_pred:
    :param y_true: Label image / ground truth [batch size, height, width].
        :type y_true:
    :param num_classes: Number of classes to predict.
        :type num_classes: int
    :return:
    """

    y_true_one_hot = nn.functional.one_hot(y_true, num_classes).float()
    y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)
    y_pred_softmax = F.softmax(y_pred, dim=1)
    dice_score = 0

    # Crossentropy Loss
    loss_ce = nn.CrossEntropyLoss()
    ce_loss = loss_ce(y_pred, y_true)

    # Channel-wise Dice loss
    for index in range(1, num_classes):
        dice_score += index * dice_loss(y_pred_softmax[:, index, :, :], y_true_one_hot[:, index, :, :],
                                        use_sigmoid=False)

    return ce_loss + 0.5 * dice_score