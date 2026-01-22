import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)



#Validation Accuracy

def evaluate_validation_accuracy(model, val_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for clips, labels in val_loader:
            clips = clips.to(device)
            labels = labels.to(device)

            outputs = model(clips)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    return all_labels, all_preds, all_probs



#Detailed Evaluation Report & Plots

def generate_evaluation_report(all_labels, all_preds, all_probs):
    """
    Generates classification report, confusion matrix,
    ROC curve, and Precision-Recall curve.
    """

    print("\n--- Detailed Classification Report ---")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["REAL", "FAKE"],
            zero_division=0
        )
    )

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REAL", "FAKE"]
    )
    disp.plot(cmap=plt.cm.Blues, ax=ax[0], colorbar=False)
    ax[0].set_title("Confusion Matrix")

 
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = roc_auc_score(all_labels, all_probs)
    ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
    ax[1].plot([0, 1], [0, 1], lw=2, linestyle="--")
    ax[1].set_xlabel("False Positive Rate (FPR)")
    ax[1].set_ylabel("True Positive Rate (TPR)")
    ax[1].set_title("ROC Curve")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=0.3)

   
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    ax[2].plot(recall, precision, lw=2, label=f"AP = {avg_precision:.3f}")
    ax[2].set_xlabel("Recall")
    ax[2].set_ylabel("Precision")
    ax[2].set_title("Precision-Recall Curve")
    ax[2].legend(loc="lower left")
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Model evaluation")