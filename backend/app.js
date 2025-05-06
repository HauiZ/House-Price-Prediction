const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const app = express();

// Enable CORS for all routes
app.use(cors());
app.use(express.json());

// API endpoint for prediction
app.post('/api/predict', (req, res) => {
    try {
        const features = req.body;
        console.log('Received prediction request:', features);
        
        // Validate required inputs
        const requiredFields = [
            'area', 'frontage', 'accessRoad', 'floors', 
            'bedrooms', 'bathrooms', 'houseDirection', 
            'balconyDirection', 'legalStatus', 'furnitureState'
        ];
        
        for (const field of requiredFields) {
            if (features[field] === undefined) {
                return res.status(400).json({ 
                    error: `Missing required field: ${field}` 
                });
            }
        }
        
        // Prepare inputs for the Python model in the correct order
        // Basic numeric features first
        const modelInputs = [
            parseFloat(features.area),
            parseFloat(features.frontage),
            parseFloat(features.accessRoad),
            parseInt(features.floors),
            parseInt(features.bedrooms),
            parseInt(features.bathrooms)
        ];
        
        // Add one-hot encoded categorical features
        // House direction
        const directions = ["Không xác định", "Bắc", "Nam", "Đông", "Tây", "Đông Bắc", "Tây Bắc", "Đông Nam", "Tây Nam"];
        directions.forEach(dir => {
            modelInputs.push(features.houseDirection === dir ? 1 : 0);
        });
        
        // Balcony direction
        directions.forEach(dir => {
            modelInputs.push((features.balconyDirection || features.houseDirection) === dir ? 1 : 0);
        });
        
        // Legal status
        const legalStatuses = ["Sổ đỏ", "Sổ hồng", "Giấy tờ hợp lệ", "Đang chờ sổ", "Khác"];
        legalStatuses.forEach(status => {
            modelInputs.push(features.legalStatus === status ? 1 : 0);
        });
        
        // Furniture state
        const furnitureStates = ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Cao cấp"];
        furnitureStates.forEach(state => {
            modelInputs.push(features.furnitureState === state ? 1 : 0);
        });
        
        console.log('Sending to model:', modelInputs);
        
        // Verify the number of features
        const expectedFeatures = 6 + directions.length * 2 + legalStatuses.length + furnitureStates.length;
        if (modelInputs.length !== expectedFeatures) {
            throw new Error(`Invalid number of features. Expected ${expectedFeatures}, got ${modelInputs.length}`);
        }
        
        // Spawn Python process with the model inputs
        const pythonProcess = spawn('python', [
            path.join(__dirname, '../model/serve.py'),
            ...modelInputs.map(String)
        ]);
        
        let predictionData = '';
        let errorData = '';
        
        pythonProcess.stdout.on('data', (data) => {
            predictionData += data.toString();
        });
        
        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            console.error(`Python Error: ${data}`);
        });
        
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                console.error(`Python process exited with code ${code}`);
                return res.status(500).json({ 
                    error: 'Model prediction failed', 
                    details: errorData 
                });
            }
            
            try {
                const prediction = parseFloat(predictionData.trim());
                if (isNaN(prediction)) {
                    throw new Error('Invalid prediction result');
                }
                
                // Convert prediction to VND (billions)
                const finalPrediction = prediction * 1000000000; // Convert billions to VND
                
                res.json({ 
                    prediction: finalPrediction,
                    predictionInBillions: prediction,
                    modelInputs: modelInputs // Useful for debugging
                });
            } catch (err) {
                res.status(500).json({ 
                    error: 'Failed to parse prediction result',
                    raw: predictionData,
                    details: err.message
                });
            }
        });
        
    } catch (err) {
        console.error('Server error:', err);
        res.status(500).json({ error: err.message });
    }
});

// Simple healthcheck endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'OK' });
});

// Add additional endpoint for fetching location metadata
app.get('/api/locations', (req, res) => {
    // This would typically come from a database
    // Here we're just providing some sample data for HCMC districts
    const locations = [
        { name: "Quận 1", priceMultiplier: 1.5 },
        { name: "Quận 2", priceMultiplier: 1.3 },
        { name: "Quận 3", priceMultiplier: 1.4 },
        { name: "Quận 4", priceMultiplier: 1.2 },
        { name: "Quận 5", priceMultiplier: 1.25 },
        { name: "Quận 6", priceMultiplier: 1.1 },
        { name: "Quận 7", priceMultiplier: 1.35 },
        { name: "Quận 8", priceMultiplier: 1.0 },
        { name: "Quận 9", priceMultiplier: 1.15 },
        { name: "Quận 10", priceMultiplier: 1.3 },
        { name: "Quận 11", priceMultiplier: 1.2 },
        { name: "Quận 12", priceMultiplier: 1.0 },
        { name: "Quận Bình Thạnh", priceMultiplier: 1.3 },
        { name: "Quận Tân Bình", priceMultiplier: 1.2 },
        { name: "Quận Tân Phú", priceMultiplier: 1.1 },
        { name: "Quận Phú Nhuận", priceMultiplier: 1.35 },
        { name: "Quận Gò Vấp", priceMultiplier: 1.15 },
        { name: "Quận Bình Tân", priceMultiplier: 1.05 },
        { name: "Huyện Củ Chi", priceMultiplier: 0.8 },
        { name: "Huyện Hóc Môn", priceMultiplier: 0.85 },
        { name: "Huyện Bình Chánh", priceMultiplier: 0.9 },
        { name: "Huyện Nhà Bè", priceMultiplier: 0.95 },
        { name: "Huyện Cần Giờ", priceMultiplier: 0.75 }
    ];
    
    res.json(locations);
});

// Serve static files from the client build directory in production
if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, '../client/build')));
    app.get('*', (req, res) => {
        res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
    });
}

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));