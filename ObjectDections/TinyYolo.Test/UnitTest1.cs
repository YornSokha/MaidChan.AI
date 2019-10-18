using System;
using System.Drawing;
using System.IO;
using System.Linq;
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
      var onnxPath = @"../../../../TinyYolo/Data/TinyYolo2_model.onnx";
      var testDataPath = @"../../../Data";

      var ml = new MLContext();
      var train = new Train(ml);

      var yolo = new TinyYoloModel(CommonHelper.GetAbsolutionPath(onnxPath, typeof(TinyYoloModel)));
      var transformer = train.LoadOnnx(yolo);

      var testImage = Path.Combine(testDataPath, "Image1.jpg" /** Image file here */);
      testImage = CommonHelper.GetAbsolutionPath(testImage, typeof(UnitTest1));

      var image = Image.FromFile(testImage);
      var bitmap = (Bitmap)image;
      var testImageData = new ImageNetData { Image = bitmap };

      var objectDetection = ml.Model.CreatePredictionEngine<ImageNetData, TinyYoloPrediction>(transformer);

      TinyYoloPrediction result = new TinyYoloPrediction();
      objectDetection.Predict(testImageData, ref result);

      OnnxOutputParser parser = new OnnxOutputParser(yolo);
      var boxes = parser.ParseOutputs(result.PredictedLabels);
      boxes = parser.FilterBoundingBoxes(boxes, 3, .5f);

      if (boxes.Any())
      {
        var saveLocation = CommonHelper.GetAbsolutionPath(testDataPath + "/result", typeof(UnitTest1));
        CommonHelper.DrawRect(testDataPath, saveLocation, "Image1.jpg", boxes);
      }
      else
      {
        Assert.False(true);
      }
    }
  }
}
