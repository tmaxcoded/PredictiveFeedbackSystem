using System;
using Microsoft.ML;
using System.Collections.Generic;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;

namespace PredictiveFeedbackSystem
{
    class Program
    {
        static List<FeedBackTrainingData> trainingdata = new List<FeedBackTrainingData>();

        static List<FeedBackTrainingData> testdata = new List<FeedBackTrainingData>();


        static void Main(string[] args)
        {
            // step 1 :- We need to ;oad the raining Data
            LoadTrainingData();

            // step 2 :- create Object of MlContext

            var mlContext = new MLContext();

            //step 3 :- convert your data to IDataView

            IDataView dataView = mlContext.CreateStreamingDataView<FeedBackTrainingData>(trainingdata);

            //staep4 : - We need to create the pipe line 
            //define the wrok flow in it.

            var pipeline = mlContext.Transforms.Text.FeaturizeText("FeedBackText", "Features")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves:50,numTrees:50,minDatapointsInLeaves:1));

            // step5:- Train the algorithm and we want the model out
            var model = pipeline.Fit(dataView);

            // step6:- Load the test data and run the test data
            // to check our models accuracy
            LoadTestData();
            IDataView dataview1 = mlContext.CreateStreamingDataView<FeedBackTrainingData>(testdata);
            var predictions = model.Transform(dataview1);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine(metrics.Accuracy);
            Console.ReadLine();


            string strcont = "Y";

            while(strcont == "Y") { 
            Console.WriteLine("Enter a feedback string");

            // step 7 :- usw model
            string feedbackstring = Console.ReadLine().ToString();
            var predictionFunction = model.MakePredictionFunction<FeedBackTrainingData, FeedBackPrediction>(mlContext);

            var feedbackInput = new FeedBackTrainingData();
            feedbackInput.FeedBackText = feedbackstring;

            var feedbacPredict = predictionFunction.Predict(feedbackInput);
            Console.WriteLine("Predicted :- " + feedbacPredict.IsGood);

            }
            Console.ReadLine();
    
            //Console.WriteLine("Hello World!");
        }

        private static void LoadTrainingData()
        {
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is good",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is horrible",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "average and ok",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad and hell",
                IsGood = false
            });
            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is nice but better can be done",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "bad bad",
                IsGood = false
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "till now it looks nice",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "quiet average",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice and good",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Good Horible",
                IsGood = false
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Cool Nice",
                IsGood = true
            });

            trainingdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Badly nice",
                IsGood = false
            });
        }

        private static void LoadTestData()
        {
            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "good",
                IsGood = true
            });


            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "horrible terrible",
                IsGood = false
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "nice",
                IsGood = true
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Bad",
                IsGood = false
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "hell",
                IsGood = false
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "sweet",
                IsGood = true
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "this is nice",
                IsGood = true
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "average",
                IsGood = true
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "ok",
                IsGood = true
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Horrible",
                IsGood = false
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Cool",
                IsGood = true
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Badly",
                IsGood = false
            });

            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Bad",
                IsGood = false
            });


            testdata.Add(new FeedBackTrainingData()
            {
                FeedBackText = "Looks nice",
                IsGood = true
            });
        }
    }
}
