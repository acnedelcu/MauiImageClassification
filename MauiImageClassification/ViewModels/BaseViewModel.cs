using CommunityToolkit.Mvvm.ComponentModel;

namespace MauiImageClassification.ViewModels
{
    public partial class BaseViewModel: ObservableObject
    {
        /// <summary>
        /// Constructor
        /// </summary>
        public BaseViewModel()
        {
            
        }

        [ObservableProperty]
        [NotifyPropertyChangedFor(nameof(IsNotBusy))]
        private bool isBusy;

        [ObservableProperty]
        string title;

        public bool IsNotBusy => !IsBusy;
    }
}
