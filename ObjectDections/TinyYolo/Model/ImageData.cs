
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace MaidChan.AI.TinyYolo.Model
{

  public struct ImageNetSettings
  {
    public const int Width = 416;
    public const int Height = 416;
  }

  public class ImageNetData
  {
    [ImageType(ImageNetSettings.Width, ImageNetSettings.Height)]
    public Bitmap Image { get; set; }
  }
}