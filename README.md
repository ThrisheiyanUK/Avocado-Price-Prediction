# 🥑 Avocado Price Predictor

A beautiful, modern web application for predicting avocado prices using machine learning, featuring stunning animations and an intuitive user interface.

## ✨ Features

### 🎨 Modern Design
- **Avocado-themed UI**: Beautiful green gradient design with avocado emojis and animations
- **Responsive Layout**: Fully responsive design that works on all devices
- **Glass Morphism**: Modern frosted glass effect on form elements
- **Custom Scrollbar**: Themed scrollbar matching the avocado color palette

### 🎭 Smooth Animations (Anime.js)
- **Page Load Animations**: Smooth entrance animations for all elements
- **Form Interactions**: Hover effects, focus animations, and micro-interactions
- **Loading States**: Beautiful loading overlays with spinning animations
- **Celebration Effects**: Floating avocado emojis when predictions are successful
- **Background Animations**: Subtle floating elements and moving gradients
- **Validation Feedback**: Shake animations for invalid inputs

### 🤖 Smart Features
- **Form Validation**: Real-time validation with animated error messages
- **Loading Indicators**: Professional loading states during prediction
- **Error Handling**: Graceful error handling with user-friendly messages
- **Auto-hide Messages**: Messages automatically disappear after a few seconds

### 🧠 Machine Learning
- **Random Forest Model**: Advanced ML algorithm for accurate predictions
- **Multiple Factors**: Considers date, region, volume, and bag sizes
- **Real-time Predictions**: Instant results based on user input

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- joblib

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd "E:\Projects\Avacado price prediction"
   ```

2. **Install required packages:**
   ```bash
   pip install flask pandas numpy scikit-learn joblib
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Open your browser and visit:**
   ```
   http://localhost:5000
   ```

## 📁 Project Structure

```
Avacado price prediction/
├── app.py                      # Flask application
├── templates/
│   └── index.html             # Enhanced HTML template
├── static/
│   ├── css/
│   │   └── style.css          # Custom CSS with animations
│   └── js/
│       └── app.js             # Anime.js animations and interactions
├── random_forest_model.pkl    # Trained ML model
├── region_encoder.pkl         # Region encoder
├── avocado.csv               # Dataset
├── Final.csv                 # Processed dataset
├── Mid.csv                   # Intermediate dataset
├── model.py                  # Model training script
└── README.md                 # This file
```

## 🎯 How to Use

1. **Select Date**: Choose the date for prediction
2. **Choose Region**: Select from 50+ available regions
3. **Enter Volume Data**: Input total volume and bag counts
4. **Predict**: Click the prediction button and enjoy the animations!
5. **View Results**: Get instant price predictions with celebration effects

## 🎨 Design Features

### Color Palette
- **Primary Green**: `#8BC34A` (Light Avocado Green)
- **Dark Green**: `#689F38` (Deep Avocado Green)  
- **Accent Green**: `#4CAF50` (Bright Green)
- **Background**: Gradient from light to medium green tones

### Animations
- **Entrance**: Elements fade and slide in smoothly
- **Interactions**: Buttons and inputs respond with hover effects
- **Loading**: Professional loading states with spinners
- **Success**: Celebrating floating emojis for successful predictions
- **Validation**: Shake effects for form validation errors

### Responsive Design
- **Mobile-First**: Optimized for mobile devices
- **Tablet Support**: Beautiful layout on tablets
- **Desktop**: Full-featured experience on desktop

## 🔧 Customization

### Adding New Animations
Edit `static/js/app.js` to add new Anime.js animations:

```javascript
anime({
    targets: '.your-element',
    translateY: [0, -10],
    duration: 300,
    easing: 'easeOutCubic'
});
```

### Styling Changes
Modify `static/css/style.css` to customize colors, layouts, and effects.

### Adding New Regions
Update the region mapping in `app.py` to support additional regions.

## 🤖 Model Information

- **Algorithm**: Random Forest Regressor
- **Features**: Date components, volume data, bag sizes, region
- **Training Data**: Historical avocado price data
- **Accuracy**: Optimized for real-world price prediction scenarios

## 📱 Browser Support

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Mobile browsers

## 🎉 Credits

- **Anime.js**: Animation library for smooth transitions
- **Google Fonts**: Poppins font family
- **Flask**: Web framework
- **scikit-learn**: Machine learning library

## 📄 License

This project is open source and available under the MIT License.

---

**Enjoy predicting avocado prices with style! 🥑✨**
