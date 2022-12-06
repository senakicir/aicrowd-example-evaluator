import pandas as pd
import numpy as np
import torch

class AIcrowdEvaluator:
  def __init__(self, ground_truth_path, **kwargs):
    """
    This is the AIcrowd evaluator class which will be used for the evaluation.
    Please note that the class name should be `AIcrowdEvaluator`
    `ground_truth` : Holds the path for the ground truth which is used to score the submissions.
    """
    self.ground_truth_path = ground_truth_path

  def _evaluate(self, client_payload, _context={}):
    """
    `client_payload` will be a dict with (atleast) the following keys :
      - submission_file_path : local file path of the submitted file
      - aicrowd_submission_id : A unique id representing the submission
      - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
    """
    submission_file_path = client_payload["submission_file_path"]
    aicrowd_submission_id = client_payload["aicrowd_submission_id"]
    aicrowd_participant_uid = client_payload["aicrowd_participant_id"]
    
    submission = torch.load(submission_file_path).numpy()
    ground_truth = np.load(self.ground_truth_path)
    # Or your preferred way to read your submission

    """
    Do something with your submitted file to come up
    with a score and a secondary score.

    If you want to report back an error to the user,
    then you can simply do :
      `raise Exception("YOUR-CUSTOM-ERROR")`

     You are encouraged to add as many validations as possible
     to provide meaningful feedback to your users
    """

    def accuracy_fn(pred_labels, gt_labels):
      return np.mean(pred_labels == gt_labels)*100

    def macrof1_fn(pred_labels,gt_labels):
      class_ids = np.unique(gt_labels)
      macrof1 = 0
      for val in class_ids:
          predpos = (pred_labels == val)
          gtpos = (gt_labels==val)
          
          tp = sum(predpos*gtpos)
          fp = sum(predpos*~gtpos)
          fn = sum(~predpos*gtpos)
          if tp == 0:
              continue
          else:
              precision = tp/(tp+fp)
              recall = tp/(tp+fn)
          macrof1 += 2*(precision*recall)/(precision+recall)
      return macrof1/len(class_ids)

    _result_object = {
        "accuracy": accuracy_fn(submission, ground_truth),
        "F1_score" : macrof1_fn(submission, ground_truth)
    }
    
    assert "accuracy" in _result_object
    assert "F1_score" in _result_object

    return _result_object

if __name__ == "__main__":
    # Lets assume the the ground_truth is a CSV file
    # and is present at data/ground_truth.csv
    # and a sample submission is present at data/sample_submission.csv
    ground_truth_path = "/Users/kicirogl/workspace/intro-ml-c233/archive/project/h36m_data/h36m_test2_labels.npy"
    _client_payload = {}
    _client_payload["submission_file_path"] =  "/Users/kicirogl/workspace/intro-ml-c233/archive/project/project_with_solutions/results_class.txt"
    _client_payload["aicrowd_submission_id"] = 1234
    _client_payload["aicrowd_participant_id"] = 1234
    
    # Instaiate a dummy context
    _context = {}

    # Instantiate an evaluator
    aicrowd_evaluator = AIcrowdEvaluator(ground_truth_path)
    
    # Evaluate
    result = aicrowd_evaluator._evaluate(_client_payload, _context)
    print(result)
