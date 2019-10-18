
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using MaidChan.AI.TinyYolo;
using MaidChan.AI.TinyYolo.Model;

namespace TinyYolo
{
  public class CommonHelper
  {
    public static void DrawRect(string inputImageLocation, string outputImageLocation, string imageName, List<BoundingBox> boundingBoxes)
    {
      Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));
      var originWidth = image.Width;
      var originHeigh = image.Height;

      foreach (var box in boundingBoxes)
      {
        // Get box dimensions
        var x = (uint)Math.Max(box.Dimensions.X, 0);
        var y = (uint)Math.Max(box.Dimensions.Y, 0);
        var width = (uint)Math.Min(originWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originWidth - y, box.Dimensions.Height);


        // Resize To Image
        x = (uint)originWidth * x / ImageNetSettings.Width;
        y = (uint)originHeigh * y / ImageNetSettings.Height;
        width = (uint)originWidth * width / ImageNetSettings.Width;
        height = (uint)originHeigh * height / ImageNetSettings.Height;

        // Prediction text
        string text = $"{box.Label} ({(box.Confidence * 100)}%)";

        using (Graphics g = Graphics.FromImage(image))
        {
          g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
          g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
          g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

          Font font = new Font("Roboto Mono", 13, FontStyle.Bold);
          SizeF textSize = g.MeasureString(text, font);
          SolidBrush brush = new SolidBrush(Color.Black);
          Point atPoint = new Point((int)x, (int)y - (int)textSize.Height - 1);

          Pen pen = new Pen(box.BoxColor, 3.2f);
          SolidBrush boxBrush = new SolidBrush(box.BoxColor);
          g.DrawString(text, font, brush, atPoint);

          g.DrawRectangle(pen, x, y, width, height);

          if (!Directory.Exists(outputImageLocation))
          {
            Directory.CreateDirectory(outputImageLocation);
          }

          image.Save(Path.Combine(outputImageLocation, imageName));
        }
      }

    }

    public static string GetAbsolutionPath(string relativePath, Type type)
    {
      FileInfo info = new FileInfo(type.Assembly.Location);
      string assembyPath = info.Directory.FullName;

      return Path.Combine(assembyPath, relativePath);
    }
  }
}