import numpy as np

def evaluate_predictions(y_pred, y_true, print_statistics=True, convert_predictions=True):
    num_classes = y_true.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        if convert_predictions:
            one_hot_pred = np.zeros((len(y_pred), num_classes))
            one_hot_pred[np.arange(len(y_pred)), y_pred] = 1

            one_hot_gt = y_true
        else:
            one_hot_pred = y_pred
            one_hot_gt = y_true

        actually_there = (np.sum(one_hot_gt, axis=0) > 0).astype("int32")

        tp = np.sum(one_hot_pred * one_hot_gt, axis=0)
        fp = np.sum(one_hot_pred * (1-one_hot_gt), axis=0)
        tn = np.sum((1-one_hot_pred) * (1-one_hot_gt), axis=0)
        fn = np.sum((1-one_hot_pred) * one_hot_gt, axis=0)

        precision = tp / (tp + fp)
        recall = tp / (fn + tp)
        f1 = 2 * precision * recall / (precision + recall)
        precision[np.isnan(precision)] = 0
        recall[np.isnan(recall)] = 0
        f1[np.isnan(f1)] = 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        if print_statistics:
            print("\nPer class metrics:")
            print("  Class | Precision  |  Recall  |    F1   |  Accuracy")
            for c in range(num_classes):
                if actually_there[c] > 0:
                    print(f"   {c:2d}   |   {precision[c]:.3f}    |  {recall[c]:.3f}   |  {f1[c]:.3f}  |   {accuracy[c]:.3f}")
                else:
                    print(f"    -   |     -      |    -     |    -    |     -   ")


        macro_f1 = np.sum(f1*actually_there) / np.sum(actually_there)
        if print_statistics:
            print(f"Macro F1: {macro_f1}")

    return macro_f1