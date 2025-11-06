// Titanic Binary Classifier - Shallow Neural Network
// TensorFlow.js implementation running entirely in browser

// Global variables
let trainData = null;
let testData = null;
let model = null;
let processedTrainData = null;
let currentThreshold = 0.5;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Titanic Classifier App...');
    
    // Set up event listeners for range inputs
    document.getElementById('epochs').addEventListener('input', function() {
        document.getElementById('epochsValue').textContent = this.value;
    });
    
    document.getElementById('batchSize').addEventListener('input', function() {
        document.getElementById('batchValue').textContent = this.value;
    });
    
    document.getElementById('threshold').addEventListener('input', function() {
        document.getElementById('thresholdValue').textContent = this.value;
        currentThreshold = parseFloat(this.value);
    });
    
    // Set up button event listeners
    document.getElementById('loadBtn').addEventListener('click', loadData);
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('modelBtn').addEventListener('click', createModel);
    document.getElementById('trainBtn').addEventListener('click', trainModel);
    document.getElementById('stopBtn').addEventListener('click', stopTraining);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('predictBtn').addEventListener('click', predictTestData);
    document.getElementById('exportModelBtn').addEventListener('click', exportModel);
    document.getElementById('exportBtn').addEventListener('click', downloadPredictions);
    
    console.log('✅ All event listeners initialized');
    document.getElementById('dataStatus').innerHTML = '<div class="status status-info">✅ App initialized. Load your CSV files to begin.</div>';
});

// Load CSV data
async function loadData() {
    console.log('📁 Loading data...');
    const trainFile = document.getElementById('trainFile').files[0];
    const statusDiv = document.getElementById('dataStatus');
    const previewDiv = document.getElementById('dataPreview');

    if (!trainFile) {
        alert('❌ Please select train.csv file first!');
        return;
    }

    statusDiv.innerHTML = '<div class="status status-info">📥 Loading data...</div>';
    
    try {
        // Read and parse CSV file
        const text = await readFile(trainFile);
        trainData = parseCSV(text);
        
        // Try to load test file if provided
        const testFile = document.getElementById('testFile').files[0];
        if (testFile) {
            const testText = await readFile(testFile);
            testData = parseCSV(testText);
        }
        
        displayDataInfo();
        statusDiv.innerHTML = '<div class="status status-success">✅ Data loaded successfully! ' + trainData.length + ' rows loaded.</div>';
        
    } catch (error) {
        console.error('Error loading data:', error);
        statusDiv.innerHTML = '<div class="status status-error">❌ Error loading data: ' + error.message + '</div>';
    }
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('File reading failed'));
        reader.readAsText(file);
    });
}

// Parse CSV with proper comma handling
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return [];
    
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                let value = values[index].trim().replace(/"/g, '');
                // Convert to number if possible
                if (!isNaN(value) && value !== '') {
                    value = parseFloat(value);
                } else if (value === '') {
                    value = null;
                }
                row[header] = value;
            });
            data.push(row);
        }
    }
    
    return data;
}

// Parse CSV line handling quoted fields
function parseCSVLine(line) {
    const result = [];
    let inQuotes = false;
    let currentField = '';
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
            result.push(currentField);
            currentField = '';
        } else {
            currentField += char;
        }
    }
    
    result.push(currentField);
    return result;
}

// Display data information
function displayDataInfo() {
    const previewDiv = document.getElementById('dataPreview');
    
    let html = `<h3>📊 Data Overview</h3>`;
    html += `<p><strong>Train Data:</strong> ${trainData.length} rows, ${Object.keys(trainData[0]).length} columns</p>`;
    
    if (testData) {
        html += `<p><strong>Test Data:</strong> ${testData.length} rows, ${Object.keys(testData[0]).length} columns</p>`;
    }
    
    // Show first few rows
    html += `<h4>Data Preview (first 5 rows):</h4>`;
    html += `<div style="overflow-x: auto;"><table>`;
    html += `<tr>${Object.keys(trainData[0]).map(key => `<th>${key}</th>`).join('')}</tr>`;
    trainData.slice(0, 5).forEach(row => {
        html += `<tr>${Object.values(row).map(val => `<td>${val !== null ? val : 'NULL'}</td>`).join('')}</tr>`;
    });
    html += `</table></div>`;
    
    previewDiv.innerHTML = html;
}

// Preprocess the data
function preprocessData() {
    console.log('⚙️ Preprocessing data...');
    const statusDiv = document.getElementById('preprocessStatus');
    
    if (!trainData) {
        alert('❌ Please load data first!');
        return;
    }
    
    statusDiv.innerHTML = '<div class="status status-info">⚙️ Preprocessing data...</div>';
    
    try {
        // Simple preprocessing: extract features and target
        const features = [];
        const targets = [];
        
        trainData.forEach(row => {
            // Use basic features
            const featureRow = [
                row.Pclass || 3,
                row.Sex === 'male' ? 1 : 0,
                row.Age || 30,
                row.SibSp || 0,
                row.Parch || 0,
                row.Fare || 32
            ];
            
            // Add engineered features if selected
            if (document.getElementById('addFamilySize').checked) {
                featureRow.push((row.SibSp || 0) + (row.Parch || 0) + 1);
            }
            
            if (document.getElementById('addIsAlone').checked) {
                const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
                featureRow.push(familySize === 1 ? 1 : 0);
            }
            
            features.push(featureRow);
            targets.push(row.Survived || 0);
        });
        
        // Convert to tensors
        processedTrainData = {
            features: tf.tensor2d(features),
            targets: tf.tensor1d(targets),
            featureNames: ['Pclass', 'Sex_male', 'Age', 'SibSp', 'Parch', 'Fare']
        };
        
        // Add engineered feature names
        if (document.getElementById('addFamilySize').checked) {
            processedTrainData.featureNames.push('FamilySize');
        }
        if (document.getElementById('addIsAlone').checked) {
            processedTrainData.featureNames.push('IsAlone');
        }
        
        document.getElementById('featureInfo').innerHTML = `
            <h4>✅ Processed Features:</h4>
            <p><strong>Feature Count:</strong> ${processedTrainData.featureNames.length}</p>
            <p><strong>Features:</strong> ${processedTrainData.featureNames.join(', ')}</p>
            <p><strong>Data Shape:</strong> ${features.length} samples × ${features[0].length} features</p>
        `;
        
        statusDiv.innerHTML = '<div class="status status-success">✅ Data preprocessing completed!</div>';
        
    } catch (error) {
        console.error('Error preprocessing data:', error);
        statusDiv.innerHTML = '<div class="status status-error">❌ Error preprocessing data: ' + error.message + '</div>';
    }
}

// Create the neural network model
function createModel() {
    console.log('🧠 Creating model...');
    const statusDiv = document.getElementById('modelStatus');
    
    if (!processedTrainData) {
        alert('❌ Please preprocess data first!');
        return;
    }
    
    try {
        const inputShape = processedTrainData.features.shape[1];
        
        // Create shallow neural network
        model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputShape],
                    units: 16,
                    activation: 'relu',
                    name: 'hidden_layer'
                }),
                tf.layers.dense({
                    units: 1,
                    activation: 'sigmoid',
                    name: 'output_layer'
                })
            ]
        });
        
        // Compile the model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        const summaryDiv = document.getElementById('modelSummary');
        summaryDiv.innerHTML = `
            <h4>✅ Model Created Successfully!</h4>
            <p><strong>Architecture:</strong></p>
            <ul>
                <li>Input: ${inputShape} features</li>
                <li>Hidden: Dense(16, relu)</li>
                <li>Output: Dense(1, sigmoid)</li>
            </ul>
            <p><strong>Total Parameters:</strong> ${model.countParams().toLocaleString()}</p>
        `;
        
        statusDiv.innerHTML = '<div class="status status-success">✅ Model created successfully!</div>';
        
    } catch (error) {
        console.error('Error creating model:', error);
        statusDiv.innerHTML = '<div class="status status-error">❌ Error creating model: ' + error.message + '</div>';
    }
}

// Train the model
async function trainModel() {
    console.log('🎯 Training model...');
    const statusDiv = document.getElementById('trainingStatus');
    
    if (!model) {
        alert('❌ Please create model first!');
        return;
    }
    
    if (!processedTrainData) {
        alert('❌ Please preprocess data first!');
        return;
    }
    
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    
    // Update UI
    document.getElementById('trainBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    statusDiv.innerHTML = '<div class="status status-info">🎯 Training started... (This may take a moment)</div>';
    
    try {
        // Simple training without validation split for demo
        await model.fit(processedTrainData.features, processedTrainData.targets, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
                    statusDiv.innerHTML = `<div class="status status-info">🎯 Training... Epoch ${epoch + 1}/${epochs}<br>
                    Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}</div>`;
                }
            }
        });
        
        statusDiv.innerHTML = '<div class="status status-success">✅ Training completed successfully!</div>';
        
    } catch (error) {
        console.error('Error training model:', error);
        statusDiv.innerHTML = '<div class="status status-error">❌ Error training model: ' + error.message + '</div>';
    } finally {
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }
}

// Stop training
function stopTraining() {
    console.log('⏹️ Stopping training...');
    document.getElementById('trainingStatus').innerHTML = '<div class="status status-info">⏹️ Training stopped by user</div>';
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

// Evaluate the model
async function evaluateModel() {
    console.log('📊 Evaluating model...');
    const metricsDiv = document.getElementById('metricsDisplay');
    
    if (!model || !processedTrainData) {
        alert('❌ Please train the model first!');
        return;
    }
    
    try {
        // Simple evaluation using training data
        const predictions = model.predict(processedTrainData.features);
        const probs = await predictions.data();
        const targets = await processedTrainData.targets.data();
        
        predictions.dispose();
        
        // Calculate basic metrics
        let correct = 0;
        let total = targets.length;
        
        for (let i = 0; i < total; i++) {
            const pred = probs[i] > currentThreshold ? 1 : 0;
            if (pred === targets[i]) correct++;
        }
        
        const accuracy = correct / total;
        
        // Display metrics
        metricsDiv.innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${accuracy.toFixed(3)}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${total}</div>
                <div class="metric-label">Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${correct}</div>
                <div class="metric-label">Correct</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(total - correct)}</div>
                <div class="metric-label">Incorrect</div>
            </div>
        `;
        
        // Show confusion matrix
        displayConfusionMatrix(targets, probs);
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        alert('❌ Error evaluating model: ' + error.message);
    }
}

// Display confusion matrix
function displayConfusionMatrix(trueLabels, probabilities) {
    const confusionDiv = document.getElementById('confusionMatrix');
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (let i = 0; i < trueLabels.length; i++) {
        const pred = probabilities[i] > currentThreshold ? 1 : 0;
        const actual = trueLabels[i];
        
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 0 && actual === 1) fn++;
    }
    
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    
    confusionDiv.innerHTML = `
        <h4>📈 Confusion Matrix (Threshold: ${currentThreshold})</h4>
        <table style="width: auto; margin: 10px 0;">
            <tr>
                <th></th>
                <th>Predicted 0</th>
                <th>Predicted 1</th>
            </tr>
            <tr>
                <th>Actual 0</th>
                <td style="background: #d4edda;">${tn} (TN)</td>
                <td style="background: #f8d7da;">${fp} (FP)</td>
            </tr>
            <tr>
                <th>Actual 1</th>
                <td style="background: #f8d7da;">${fn} (FN)</td>
                <td style="background: #d4edda;">${tp} (TP)</td>
            </tr>
        </table>
        <p><strong>Precision:</strong> ${precision.toFixed(3)} | <strong>Recall:</strong> ${recall.toFixed(3)} | <strong>F1-Score:</strong> ${f1.toFixed(3)}</p>
    `;
}

// Predict on test data
async function predictTestData() {
    console.log('🔮 Predicting test data...');
    const statusDiv = document.getElementById('predictionStatus');
    
    if (!model) {
        alert('❌ Please train the model first!');
        return;
    }
    
    if (!testData) {
        alert('❌ Please load test.csv file first!');
        return;
    }
    
    statusDiv.innerHTML = '<div class="status status-info">🔮 Making predictions...</div>';
    
    try {
        // Simple prediction logic
        const predictions = [];
        
        for (let row of testData) {
            const features = [
                row.Pclass || 3,
                row.Sex === 'male' ? 1 : 0,
                row.Age || 30,
                row.SibSp || 0,
                row.Parch || 0,
                row.Fare || 32
            ];
            
            // Add engineered features if they were used in training
            if (processedTrainData.featureNames.includes('FamilySize')) {
                features.push((row.SibSp || 0) + (row.Parch || 0) + 1);
            }
            
            if (processedTrainData.featureNames.includes('IsAlone')) {
                const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
                features.push(familySize === 1 ? 1 : 0);
            }
            
            const input = tf.tensor2d([features]);
            const prediction = model.predict(input);
            const prob = await prediction.data();
            
            predictions.push({
                PassengerId: row.PassengerId,
                Survived: prob[0] > currentThreshold ? 1 : 0,
                Probability: prob[0]
            });
            
            // Clean up tensors
            input.dispose();
            prediction.dispose();
        }
        
        testPredictions = predictions;
        
        statusDiv.innerHTML = `
            <div class="status status-success">
                ✅ Predictions completed!<br>
                ${predictions.length} predictions made.<br>
                ${predictions.filter(p => p.Survived === 1).length} passengers predicted to survive.
            </div>
        `;
        
    } catch (error) {
        console.error('Error predicting:', error);
        statusDiv.innerHTML = '<div class="status status-error">❌ Error making predictions: ' + error.message + '</div>';
    }
}

// Export model
async function exportModel() {
    if (!model) {
        alert('❌ Please train the model first!');
        return;
    }
    
    try {
        await model.save('downloads://titanic-model');
        document.getElementById('exportInfo').innerHTML = '<div class="status status-success">✅ Model exported successfully!</div>';
    } catch (error) {
        console.error('Error exporting model:', error);
        document.getElementById('exportInfo').innerHTML = '<div class="status status-error">❌ Error exporting model: ' + error.message + '</div>';
    }
}

// Download predictions
function downloadPredictions() {
    if (!testPredictions) {
        alert('❌ Please make predictions first!');
        return;
    }
    
    try {
        // Create CSV content
        let csvContent = 'PassengerId,Survived,Probability\n';
        testPredictions.forEach(pred => {
            csvContent += `${pred.PassengerId},${pred.Survived},${pred.Probability.toFixed(4)}\n`;
        });
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'titanic_predictions.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        document.getElementById('exportInfo').innerHTML = '<div class="status status-success">✅ Predictions downloaded as CSV!</div>';
        
    } catch (error) {
        console.error('Error downloading predictions:', error);
        document.getElementById('exportInfo').innerHTML = '<div class="status status-error">❌ Error downloading predictions: ' + error.message + '</div>';
    }
}

console.log('🎉 Titanic Binary Classifier App Loaded Successfully!');
