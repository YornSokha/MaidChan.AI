
using Microsoft.ML.Data;

namespace MaidChan.AI.TinyYolo.Model
{
  public class TinyYoloModel : IOnnxModel
  {
    public string ModelPath { get; private set; }

    public string InputName { get; } = "image";

    public string OutputName { get; } = "grid";

    public string[] Labels { get; } = {
      "aeroplane", "bicycle", "bird", "boat", "bottle",
      "bus", "car", "cat", "chair", "cow",
      "diningtable", "dog", "horse", "motorbike", "person",
      "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };

    public (float, float)[] Anchors { get; set; } = { (1.08f, 1.19f), (3.42f, 4.41f), (6.63f, 11.38f), (9.42f, 5.11f), (16.62f, 10.52f) };


    public TinyYoloModel(string modelPath)
    {
      this.ModelPath = modelPath;
    }
  }

  public class TinyYoloPrediction : IOnnxPrediction
  {
    [ColumnName("grid")]
    public float[] PredictedLabels { get; set; }
  }
}