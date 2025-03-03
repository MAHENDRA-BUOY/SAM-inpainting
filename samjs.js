import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [resultImage, setResultImage] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!image) {
      alert('Please upload an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);
    formData.append('text_prompt', textPrompt);

    try {
      const response = await fetch('http://localhost:5000/inpaint', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResultImage(data.result_image);
    } catch (error) {
      console.error('Error processing image:', error);
    }
  };

  return (
    <div className="App">
      <h1>Segment Anything Inpainting</h1>
      <input type="file" onChange={handleImageUpload} accept="image/*" />
      <input
        type="text"
        placeholder="Enter text prompt (optional)"
        value={textPrompt}
        onChange={(e) => setTextPrompt(e.target.value)}
      />
      <button onClick={handleSubmit}>Process Image</button>

      {image && <img src={image} alt="Uploaded" className="preview" />}
      {resultImage && <img src={resultImage} alt="Result" className="result" />}
    </div>
  );
}

export default App;
