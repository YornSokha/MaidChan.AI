
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

    public (float, float)[] Anchors { get; set; }


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