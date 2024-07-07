using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MauiImageClassification.Services
{
    public class ImageClassifier : IImageClassifier
    {
        #region PrivateMembers

        private const int DimBatchSize = 1;
        private const int DimNumberOfChannels = 3;
        private const int TargetWidth = 224;
        private const int TargetHeight = 224;

        private List<string> labels;
        private InferenceSession session;

        private const string ModelInputName = "input.1";
        private const string ModelOutputName = "650";

        #endregion

        #region Constructors

        /// <summary>
        /// Constructor
        /// </summary>
        public ImageClassifier()
        {
            this.labels = ReadLabels();
            this.session = ReadModel();
        }

        #endregion Constructors

        #region InterfaceMembers

        /// <summary>
        /// Classifies the provided image
        /// </summary>
        /// <param name="image">The image to classify</param>
        /// <returns>The classification result</returns>
        /// <exception cref="NotImplementedException"></exception>
        public async Task<string> ClassifyImageAsync(byte[] image)
        {
            if (image == null || image.Length == 0)
            {
                throw new ArgumentNullException(nameof(image));
            }

            // resize, transform and classify the image
            byte[] resizedImage = ResizeImage(image);
            float[] transformedImage = TransformImage(resizedImage);
            string classification = GetClassification(transformedImage);

            // format the output value
            classification = classification.Replace("\r", "");
            if (string.IsNullOrWhiteSpace(classification))
            {
                throw new ArgumentNullException(nameof(classification));
            }
            return classification;
        }

        #endregion

        #region PrivateMethods

        /// <summary>
        /// Classifies the normalized image
        /// </summary>
        /// <param name="normalizedImageData">The normalized image to be classified</param>
        /// <returns>The classification</returns>
        private string GetClassification(float[] normalizedImageData)
        {
            var imageTensor = new DenseTensor<float>(normalizedImageData, new[]
                {
                DimBatchSize,
                DimNumberOfChannels,
                TargetWidth,
                TargetHeight
            });

            using var results = this.session.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(ModelInputName, imageTensor)
            });

            var output = results.FirstOrDefault(i => i.Name == ModelOutputName);
            _ = output ?? throw new ArgumentNullException(nameof(output));

            List<float> scores = output.AsTensor<float>().ToList();
            float highestScore = scores.Max();
            int highestScoreIndex = scores.IndexOf(highestScore);
            string label = this.labels.ElementAt(highestScoreIndex);

            label = label.Replace("_", " ").Trim();
            return label;
        }

        /// <summary>
        /// Resizes the image to the dimensions specified by <see cref="TargetWidth"/> and <see cref="TargetHeight"/>
        /// </summary>
        /// <param name="sourceImage">The original image</param>
        /// <returns>The resized image</returns>
        private byte[] ResizeImage(byte[] sourceImage)
        {
            using SKBitmap sourceBitmap = SKBitmap.Decode(sourceImage);

            if (sourceBitmap.Width == TargetWidth && sourceBitmap.Height == TargetHeight)
            {
                return sourceImage;
            }

            float ratio = (float)Math.Min(TargetWidth, TargetHeight) / Math.Min(sourceBitmap.Width, sourceBitmap.Height);

            SKImageInfo info = new((int)(ratio * sourceBitmap.Width), (int)(ratio * sourceBitmap.Height));
            using SKBitmap scaledBitmap = sourceBitmap.Resize(info, SKFilterQuality.Medium);

            int horizontalCrop = scaledBitmap.Width - TargetWidth;
            int verticalCrop = scaledBitmap.Height - TargetHeight;
            int leftOffset = horizontalCrop == 0 ? 0 : horizontalCrop / 2;
            int topOffset = verticalCrop == 0 ? 0 : verticalCrop / 2;
            var cropRect = SKRectI.Create(new SKPointI(leftOffset, topOffset), new SKSizeI(TargetWidth, TargetHeight));

            using SKImage croppedImage = SKImage.FromBitmap(scaledBitmap).Subset(cropRect);

            return croppedImage.Encode(SKEncodedImageFormat.Jpeg, 100).ToArray();
        }

        /// <summary>
        /// Read the labels from the file
        /// </summary>
        public List<string> ReadLabels()
        {
            // classes.txt must have the 'Build Action' property set to 'Embedded resource'
            var assembly = GetType().Assembly;
            using var labelsStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.classes.txt");

            if (labelsStream == null || labelsStream.Length == 0)
            {
                throw new ArgumentNullException(nameof(labelsStream));
            }

            using var reader = new StreamReader(labelsStream);

            string text = reader.ReadToEnd();
            List<string> labels = text.Split(new string[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries).ToList();
            return labels;
        }

        /// <summary>
        /// Returns a new <see cref="InferenceSession"/> using the .onnx file
        /// </summary>
        private InferenceSession ReadModel()
        {
            // the onnx file must have the 'Build Action' property set to 'Embedded resource'
            var assembly = GetType().Assembly;
            using Stream? modelStream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.efficientnet_food101model.onnx");

            if (modelStream == null || modelStream.Length == 0)
            {
                throw new ArgumentNullException(nameof(modelStream));
            }

            using MemoryStream stream = new();
            modelStream.CopyTo(stream);
            return new InferenceSession(stream.ToArray());
        }

        /// <summary>
        /// Applies the mean and std values required by the model on the image and produces a single array of values
        /// </summary>
        /// <param name="image">The RGB image</param>
        /// <returns>A single array of values representing the normalized values</returns>
        private float[] TransformImage(byte[] image)
        {
            SKBitmap bitmap = SKBitmap.Decode(image);
            var pixels = bitmap.Pixels;

            float[] rValues = new float[pixels.Length];
            float[] gValues = new float[pixels.Length];
            float[] bValues = new float[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                rValues[i] = (pixels[i].Red / 255f - 0.485f) / 0.229f;
                gValues[i] = (pixels[i].Green / 255f - 0.456f) / 0.224f;
                bValues[i] = (pixels[i].Blue / 255f - 0.406f) / 0.225f;
            }
            return rValues.Concat(gValues).Concat(bValues).ToArray();
        }

        #endregion
    }
}
