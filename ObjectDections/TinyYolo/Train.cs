using System.Collections.Generic;
using MaidChan.AI.TinyYolo.Model;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace MaidChan.AI.TinyYolo
{
  public class Train
  {
    private readonly MLContext ml;

    public Train(MLContext ml)
    {
      this.ml = ml;
    }

    public ITransformer LoadOnnx(IOnnxModel model)
    {
      var dataview = ml.Data.LoadFromEnumerable(new List<ImageNetData>());

      var estimator = ml.Transforms.ResizeImages(
        inputColumnName: nameof(ImageNetData.Image),
        outputColumnName: model.InputName,
        imageWidth: ImageNetSettings.Width,
        imageHeight: ImageNetSettings.Height,
        resizing: ResizingKind.Fill
      )
      .Append(ml.Transforms.ExtractPixels(outputColumnName: model.InputName))
      .Append(ml.Transforms.ApplyOnnxModel(model.OutputName, model.InputName, model.ModelPath, 1, true));

      var transformer = estimator.Fit(dataview);

      return transformer;
    }

    public void SaveAsMLModel(ITransformer model, string location)
    {
    }
  }

}