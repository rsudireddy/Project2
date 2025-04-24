import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

visualization_folder = "plots"
os.makedirs(visualization_folder, exist_ok=True)

def calculate_metrics(actual_values, predicted_values):
    true_pos = np.sum((predicted_values==1) & (actual_values==1))
    false_pos = np.sum((predicted_values==1) & (actual_values==0))
    false_neg = np.sum((predicted_values==0) & (actual_values==1))
    
    prec_case = 0 if true_pos+false_pos > 0 else 1
    prec_cases = {
        0: lambda: true_pos/(true_pos+false_pos),
        1: lambda: 0.0
    }
    precision = prec_cases.get(prec_case, prec_cases[0])()
    
    rec_case = 0 if true_pos+false_neg > 0 else 1
    rec_cases = {
        0: lambda: true_pos/(true_pos+false_neg),
        1: lambda: 0.0
    }
    recall = rec_cases.get(rec_case, rec_cases[0])()
    
    f1_case = 0 if precision+recall > 0 else 1
    f1_cases = {
        0: lambda: 2*precision*recall/(precision+recall),
        1: lambda: 0.0
    }
    f1 = f1_cases.get(f1_case, f1_cases[0])()
    
    return precision, recall, f1

def compute_roc_curve(actual_values, probability_scores):
    cutoffs = np.unique(probability_scores)
    true_pos_rates, false_pos_rates = [], []
    total_pos, total_neg = np.sum(actual_values==1), np.sum(actual_values==0)
    
    threshold_idx = 0
    while threshold_idx < len(cutoffs):
        threshold = cutoffs[threshold_idx]
        binary_predictions = (probability_scores >= threshold).astype(int)
        true_pos_count = np.sum((binary_predictions==1) & (actual_values==1))
        false_pos_count = np.sum((binary_predictions==1) & (actual_values==0))
        
        tpr_case = 0 if total_pos > 0 else 1
        tpr_cases = {
            0: lambda: true_pos_count/total_pos,
            1: lambda: 0
        }
        true_pos_rates.append(tpr_cases.get(tpr_case, tpr_cases[0])())
        
        fpr_case = 0 if total_neg > 0 else 1
        fpr_cases = {
            0: lambda: false_pos_count/total_neg,
            1: lambda: 0
        }
        false_pos_rates.append(fpr_cases.get(fpr_case, fpr_cases[0])())
        
        threshold_idx += 1
        
    return np.array([0] + false_pos_rates + [1]), np.array([0] + true_pos_rates + [1]), cutoffs

def calculate_auc(false_pos_rates, true_pos_rates):
    return np.trapezoid(true_pos_rates, false_pos_rates)

def plot_accuracy_heatmap(accuracy_matrix, param1_values, param2_values, param1_name, param2_name):
    plt.figure(figsize=(6,4))
    plt.imshow(accuracy_matrix, origin='lower', aspect='auto', cmap='viridis')
    plt.xticks(range(len(param2_values)), param2_values)
    plt.yticks(range(len(param1_values)), param1_values)
    plt.colorbar(label="Accuracy")
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.title("Model Accuracy Heatmap")
    plt.savefig(os.path.join(visualization_folder, "heatmap_accuracy.png"))
    plt.close()

def plot_parameter_performance(parameter_values, performance_values, param_name, metric_name):
    plt.figure()
    plt.plot(parameter_values, performance_values, marker="o")
    plt.xlabel(param_name)
    plt.ylabel(metric_name)
    plt.title(f"Effect of {param_name} on {metric_name}")
    plt.savefig(os.path.join(visualization_folder, f"{param_name.lower()}_vs_{metric_name.lower()}.png"))
    plt.close()

def plot_roc_curve(false_pos_rates, true_pos_rates, auc_score):
    plt.figure()
    plt.plot(false_pos_rates, true_pos_rates, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(visualization_folder, "roc_curve.png"))
    plt.close()

if __name__ == "__main__":
    estimator_counts = [100, 200, 500]
    learning_rates = [0.05, 0.1, 0.2]
    accuracy_matrix = np.random.rand(len(estimator_counts), len(learning_rates))
    plot_accuracy_heatmap(accuracy_matrix, estimator_counts, learning_rates, 
                         "n_estimators", "learning_rate")
    
    tree_depths = [1, 2, 3]
    accuracy_scores = [0.82, 0.87, 0.85]
    plot_parameter_performance(tree_depths, accuracy_scores, "Tree Depth", "Accuracy")
    
    fpr = np.linspace(0, 1, 100)
    tpr = np.sqrt(fpr)
    auc_score = calculate_auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, auc_score)
    
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6, 0.3, 0.7])
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    
    optimal_threshold, optimal_f1 = 0.5, 0.0
    threshold_idx = 0
    while threshold_idx < len(thresholds):
        threshold = thresholds[threshold_idx]
        thresholded_preds = (y_scores >= threshold).astype(int)
        _, _, current_f1 = calculate_metrics(y_true, thresholded_preds)
        
        if current_f1 > optimal_f1:
            optimal_f1 = current_f1
            optimal_threshold = threshold
            
        threshold_idx += 1
    
    optimal_preds = (y_scores >= optimal_threshold).astype(int)
    precision_optimal, recall_optimal, _ = calculate_metrics(y_true, optimal_preds)
    print(f"Best threshold = {optimal_threshold:.3f}, Precision: {precision_optimal:.3f}, " +
          f"Recall: {recall_optimal:.3f}, F1: {optimal_f1:.3f}")