const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const app = express();

app.use(cors());
app.use(express.json());

app.post('/api/predict', (req, res) => {
    try {
        const features = req.body;
        console.log('Received prediction request:', features);
        
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
        
        const modelInputs = [
            parseFloat(features.area),
            parseFloat(features.frontage),
            parseFloat(features.accessRoad),
            parseInt(features.floors),
            parseInt(features.bedrooms),
            parseInt(features.bathrooms)
        ];
        
        const directions = ["Không xác định", "Bắc", "Nam", "Đông", "Tây", "Đông - Bắc", "Tây - Bắc", "Đông - Nam", "Tây - Nam"];
        directions.forEach(dir => {
            modelInputs.push(features.houseDirection === dir ? 1 : 0);
        });
        
        directions.forEach(dir => {
            modelInputs.push((features.balconyDirection || features.houseDirection) === dir ? 1 : 0);
        });
        
        const legalStatusMap = {
            "Sổ đỏ": "Have certificate",
            "Sổ hồng": "Have certificate",
            "Giấy tờ hợp lệ": "Have certificate",
            "Đang chờ sổ": "Sale contract",
            "Khác": "null"
        };
        
        const pythonLegalStatuses = ["null", "Have certificate", "Sale contract"];
        pythonLegalStatuses.forEach(status => {
            const mappedStatus = legalStatusMap[features.legalStatus] || "null";
            modelInputs.push(mappedStatus === status ? 1 : 0);
        });
        
        const furnitureStateMap = {
            "Không nội thất": "null",
            "Nội thất cơ bản": "basic",
            "Đầy đủ nội thất": "full",
            "Cao cấp": "full"
        };
        
        const pythonFurnitureStates = ["null", "basic", "full"];
        pythonFurnitureStates.forEach(state => {
            const mappedState = furnitureStateMap[features.furnitureState] || "null";
            modelInputs.push(mappedState === state ? 1 : 0);
        });
        
        console.log('Sending to model:', modelInputs);
        
        const expectedFeatures = 6 + directions.length * 2 + pythonLegalStatuses.length + pythonFurnitureStates.length;
        if (modelInputs.length !== expectedFeatures) {
            throw new Error(`Invalid number of features. Expected ${expectedFeatures}, got ${modelInputs.length}`);
        }
        
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
                
                // Đổi sang tỷ
                const finalPrediction = prediction * 1000000000; 
                
                res.json({ 
                    prediction: finalPrediction,
                    predictionInBillions: prediction,
                    modelInputs: modelInputs 
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

app.get('/api/health', (req, res) => {
    res.json({ status: 'OK' });
});

if (process.env.NODE_ENV === 'production') {
    app.use(express.static(path.join(__dirname, '../client/build')));
    app.get('*', (req, res) => {
        res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
    });
}

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));