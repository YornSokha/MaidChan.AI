using System;
using System.Drawing;
using System.IO;
using MaidChan.AI.TinyYolo;
using MaidChan.AI.TinyYolo.Model;
using Microsoft.ML;
using Xunit;

namespace TinyYolo.Test
{
  public class UnitTest1
  {
    [Fact]
    public void Test1()
    {
      var onnxPath = Path.Combine("../TinyYolo/Data/TinyYolo2_model.onnx");

      var ml = new MLContext();
      var train = new Train(ml);

      var yolo = new TinyYoloModel(onnxPath);
      var transformer = train.LoadOnnx(yolo);

      var testImage = Path.Combine("Data", "Image1.jpg" /** Image file here */);
      var image = Image.FromFile(testImage);
      var bitmap = (Bitmap)image;
      var testImageData = new ImageNetData { Image = bitmap };

      var objectDetection = ml.Model.CreatePredictionEngine<ImageNetData, TinyYoloPrediction>(transformer);

      TinyYoloPrediction result = new TinyYoloPrediction();
      objectDetection.Predict(testImageData, ref result);

      OnnxOutputParser parser = new OnnxOutputParser(yolo);
      var boxes = parser.ParseOutputs(result.PredictedLabels);
      boxes = parser.FilterBoundingBoxes(boxes, 1, .05f);
      foreach (var b in boxes)
      {
        Console.WriteLine(b.Label);
      }
    }
  }
}
