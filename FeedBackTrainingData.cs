using System;
using Microsoft.ML;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;

namespace PredictiveFeedbackSystem
{
    public class FeedBackTrainingData 
    {
        [Column(ordinal: "0", name: "Label")]
        public bool IsGood { get; set; }

        [Column(ordinal: "0")]
        public string FeedBackText { get; set; }

      
    }


    public class FeedBackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool IsGood { get; set; }

      


    }
}
