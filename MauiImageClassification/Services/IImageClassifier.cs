namespace MauiImageClassification.Services
{
    public interface IImageClassifier
    {
        Task<string> ClassifyImageAsync(byte[] image);
    }
}
