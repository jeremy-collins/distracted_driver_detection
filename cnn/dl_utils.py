"""
Utilities to be used along with the deep model
"""
from typing import Union

import torch
from my_resnet import MyResNet18
from simple_net import SimpleNet
from simple_net_final import SimpleNetFinal
from multilabel_resnet import MultilabelResNet18
from my_optimal_net import OptimalNet
from torch import nn


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K classes logits[k] (where 0 <= k < K) corresponds to the
                log-odds of class `k` being the correct one.
                Shape: (batch_size, num_classes)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size)
    Returns:
        accuracy: The accuracy of the predicted logits
                   (number of correct predictions / total number of examples)
    """

    pred = torch.argmax(logits, axis=1)
    batch_accuracy = torch.sum(pred == labels).item() / labels.numel()

    return batch_accuracy


def compute_loss(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18, OptimalNet],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target labels

    Args:
    -   model: a model (which inherits from nn.Module)
    -   model_output: the raw scores output by the net
    -   target_labels: the ground truth class labels
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the loss value
    """
    
    if target_labels.dim() == 1:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.BCELoss()
    
        model_output = model_output.to(torch.float)
        target_labels = target_labels.to(torch.float)
    
    loss = loss_function(model_output, target_labels)
    
    # if is_normalize:
    #     loss = loss / model_output.size()[0]
    
    return loss

def compute_multilabel_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy given the prediction logits and the ground-truth labels

    Args:
        logits: The output of the forward pass through the model.
                for K labels logits[k] (where 0 <= k < K) corresponds to the
                log-odds of label `k` being present in the image.
                Shape: (batch_size, num_labels)
        labels: The ground truth label for each instance in the batch
                Shape: (batch_size, num_labels)
    Returns:
        accuracy: The accuracy of the predicted logits
                  (number of correct predictions / total number of labels)
    """

    binary_pred = torch.round(logits) # shape is (batch_size, num_labels)
    batch_accuracy = torch.sum(torch.eq(binary_pred, labels).to(torch.int32)) / labels.numel()
    batch_accuracy = batch_accuracy.item()

    return batch_accuracy


def save_trained_model_weights(
    model: Union[SimpleNet, SimpleNetFinal, MyResNet18, MultilabelResNet18], out_dir: str
) -> None:
    """Saves the weights of a trained model along with class name

    Args:
    -   model: The model to be saved
    -   out_dir: The path to the folder to store the save file in
    """
    class_name = model.__class__.__name__
    state_dict = model.state_dict()

    assert class_name in set(
        ["SimpleNet", "SimpleNetFinal", "MyResNet18", "MultilabelResNet18"]
    ), "Please save only supported models"

    save_dict = {"class_name": class_name, "state_dict": state_dict}
    torch.save(save_dict, f"{out_dir}/trained_{class_name}_final.pt")
