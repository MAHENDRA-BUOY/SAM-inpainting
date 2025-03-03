const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Set up file upload
const upload = multer({ dest: 'uploads/' });

// API route for inpainting
app.post('/inpaint', upload.single('image'), (req, res) => {
    const imagePath = req.file.path;
    const textPrompt = req.body.text_prompt || "";
    const outputPath = `processed_${Date.now()}.png`;
    
    // Run the Python script
    const command = `python inpaint.py --image ${imagePath} --text "${textPrompt}" --output ${outputPath}`;
    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${error.message}`);
            return res.status(500).json({ error: "Processing failed" });
        }
        
        res.json({ result_image: `http://localhost:${PORT}/${outputPath}` });
    });
});

// Serve processed images
app.use(express.static(path.join(__dirname, 'uploads')));

// Start server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
