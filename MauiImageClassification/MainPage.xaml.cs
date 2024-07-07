using MauiImageClassification.ViewModels;

namespace MauiImageClassification
{
    public partial class MainPage : ContentPage
    {
        /// <summary>
        /// Constructor
        /// </summary>
        /// <param name="viewModel">The view model of the page</param>
        public MainPage(MainPageViewModel viewModel)
        {
            InitializeComponent();
            BindingContext = viewModel; // the connection between the view and the view model
        }
    }

}
