namespace MaidChan.AI.TinyYolo.Model
{
  public interface IOnnxModel
  {
    string ModelPath { get; }
    string InputName { get; }
    string OutputName { get; }
    string[] Labels { get; }
    (float, float)[] Anchors { get; }
  }

  public interface IOnnxPrediction
  {
    float[] PredictedLabels { get; }
  }
}