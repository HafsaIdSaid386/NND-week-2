// Titanic Binary Classifier - Shallow Neural Network
// TensorFlow.js implementation running entirely in browser

// Global variables to store data and model
let trainData = null;
let testData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let testPredictions = null;
let currentThreshold = 0.5;
let featureNames = [];

// Dataset schema configuration - SWAP THIS FOR OTHER DATASETS
const DATA_SCHEMA = {
    target: 'Survived',           // Binary classification target
    features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'], // Feature columns
    identifier: 'PassengerId',    // Exclude from modeling
    categorical: ['Sex', 'Pclass', 'Embarked'], // Categorical features for one-hot encoding
    numerical: ['Age', 'SibSp', 'Parch', 'Fare'] // Numerical features for standardization
};

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing Titanic Classifier...');
    
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
        if (validationData) {
            evaluateModel();
        }
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
    
    console.log('Event listeners initialized');
});

// Load and inspect CSV data
async function loadData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    const statusDiv = document.getElementById('dataStatus');
    const previewDiv = document.getElementById('dataPreview');

    if (!trainFile) {
        alert('Please select train.csv file');
        return;
    }

    statusDiv.innerHTML = '<div class="status status-info">Loading data...</div>';
    previewDiv.innerHTML = '';

    try {
        // Load train data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data if provided
        if (testFile) {
            const testText = await readFile(testFile);
            testData = parseCSV(testText);
        }

        displayDataInfo(trainData, testData);
        
        // Create exploratory charts
        createExploratoryCharts(trainData);
        
        statusDiv.innerHTML = '<div class="status status-success">Data loaded successfully!</div>';
        
    } catch (error) {
        console.error('Error loading data:', error);
        statusDiv.innerHTML = `<div class="status status-error">Error loading data: ${error.message}</div>`;
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

// Parse CSV with proper comma handling - FIXES HOMEWORK ISSUE #1
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim());
    if (lines.length === 0) return [];
    
    const headers = parseCSVLine(lines[0]);
    const data = [];
    
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        if (values.length === headers.length) {
            const row = {};
            headers.forEach((header, index) => {
                // Convert numerical values
                const value = values[index].trim();
                if (value === '') {
                    row[header] = null;
                } else if (!isNaN(value) && value !== '') {
                    row[header] = parseFloat(value);
                } else {
                    row[header] = value;
                }
            });
            data.push(row);
        }
    }
    
    return data;
}

// Parse CSV line handling quoted fields with commas - FIXES HOMEWORK ISSUE #1
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
    return result.map(field => field.replace(/^"|"$/g, '').trim());
}

// Display data information and preview
function displayDataInfo(trainData, testData) {
    const previewDiv = document.getElementById('dataPreview');
    
    let html = `<h3>Data Overview</h3>`;
    html += `<p><strong>Train Data:</strong> ${trainData.length} rows, ${Object.keys(trainData[0]).length} columns</p>`;
    
    if (testData) {
        html += `<p><strong>Test Data:</strong> ${testData.length} rows, ${Object.keys(testData[0]).length} columns</p>`;
    }
    
    // Calculate missing values
    const missingStats = calculateMissingValues(trainData);
    html += `<h4>Missing Values in Train Data:</h4><ul>`;
    Object.entries(missingStats).forEach(([column, stats]) => {
        html += `<li>${column}: ${stats.missing} missing (${stats.percentage}%)</li>`;
    });
    html += `</ul>`;
    
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

// Calculate missing values statistics
function calculateMissingValues(data) {
    const stats = {};
    const totalRows = data.length;
    
    Object.keys(data[0]).forEach(column => {
        const missing = data.filter(row => row[column] === null || row[column] === '' || (typeof row[column] === 'number' && isNaN(row[column]))).length;
        stats[column] = {
            missing: missing,
            percentage: ((missing / totalRows) * 100).toFixed(2)
        };
    });
    
    return stats;
}

// Create exploratory data analysis charts
function createExploratoryCharts(data) {
    // Survival by Sex
    const sexSurvival = {};
    data.forEach(row => {
        if (row.Sex && row.Survived !== undefined && row.Survived !== null) {
            const key = `${row.Sex}-${row.Survived}`;
            sexSurvival[key] = (sexSurvival[key] || 0) + 1;
        }
    });
    
    // Survival by Pclass
    const classSurvival = {};
    data.forEach(row => {
        if (row.Pclass && row.Survived !== undefined && row.Survived !== null) {
            const key = `Class ${row.Pclass}-${row.Survived}`;
            classSurvival[key] = (classSurvival[key] || 0) + 1;
        }
    });
    
    // Create charts using tfjs-vis
    const surface = { name: 'Exploratory Data Analysis', tab: 'Data Analysis' };
    
    const sexData = {
        values: Object.entries(sexSurvival).map(([key, value]) => ({ x: key, y: value }))
    };
    
    const classData = {
        values: Object.entries(classSurvival).map(([key, value]) => ({ x: key, y: value }))
    };
    
    tfvis.render.barchart(surface, [sexData, classData], {
        xLabel: 'Category-Survival',
        yLabel: 'Count'
    });
}

// Preprocess the data
function preprocessData() {
    const statusDiv = document.getElementById('preprocessStatus');
    const featureDiv = document.getElementById('featureInfo');
    
    if (!trainData) {
        alert('Please load data first');
        return;
    }
    
    statusDiv.innerHTML = '<div class="status status-info">Preprocessing data...</div>';
    
    try {
        // Create a copy of train data for preprocessing
        let processedData = JSON.parse(JSON.stringify(trainData));
        
        // Add engineered features if selected
        const addFamilySize = document.getElementById('addFamilySize').checked;
        const addIsAlone = document.getElementById('addIsAlone').checked;
        
        // Reset features to original
        DATA_SCHEMA.features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'];
        DATA_SCHEMA.numerical = ['Age', 'SibSp', 'Parch', 'Fare'];
        
        if (addFamilySize) {
            processedData.forEach(row => {
                row.FamilySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            });
            DATA_SCHEMA.features.push('FamilySize');
            DATA_SCHEMA.numerical.push('FamilySize');
        }
        
        if (addIsAlone && addFamilySize) {
            processedData.forEach(row => {
                row.IsAlone = row.FamilySize === 1 ? 1 : 0;
            });
            DATA_SCHEMA.features.push('IsAlone');
            DATA_SCHEMA.numerical.push('IsAlone');
        }
        
        // Handle missing values
        processedData = handleMissingValues(processedData);
        
        // One-hot encode categorical variables
        const { features, featureNames: encodedFeatureNames } = oneHotEncode(processedData);
        featureNames = encodedFeatureNames;
        
        // Standardize numerical features
        const standardizedFeatures = standardizeNumerical(features, featureNames);
        
        // Prepare target variable
        const targets = processedData.map(row => row[DATA_SCHEMA.target] || 0);
        
        // Convert to tensors
        const featureTensor = tf.tensor2d(standardizedFeatures);
        const targetTensor = tf.tensor1d(targets);
        
        // Store processed data
        trainData = {
            raw: processedData,
            features: featureTensor,
            targets: targetTensor,
            featureNames: featureNames
        };
        
        // Display feature information
        featureDiv.innerHTML = `
            <h4>Processed Features:</h4>
            <p><strong>Feature Count:</strong> ${featureNames.length}</p>
            <p><strong>Feature Names:</strong> ${featureNames.join(', ')}</p>
            <p><strong>Tensor Shape:</strong> [${featureTensor.shape[0]}, ${featureTensor.shape[1]}]</p>
        `;
        
        statusDiv.innerHTML = '<div class="status status-success">Data preprocessing completed!</div>';
        
    } catch (error) {
        console.error('Error preprocessing data:', error);
        statusDiv.innerHTML = `<div class="status status-error">Error preprocessing data: ${error.message}</div>`;
    }
}

// Handle missing values
function handleMissingValues(data) {
    // Calculate medians and modes for imputation
    const medians = {};
    const modes = {};
    
    DATA_SCHEMA.numerical.forEach(col => {
        const values = data.map(row => row[col]).filter(val => val !== null && !isNaN(val));
        if (values.length > 0) {
            values.sort((a, b) => a - b);
            medians[col] = values[Math.floor(values.length / 2)];
        }
    });
    
    DATA_SCHEMA.categorical.forEach(col => {
        const freq = {};
        data.forEach(row => {
            if (row[col]) {
                freq[row[col]] = (freq[row[col]] || 0) + 1;
            }
        });
        modes[col] = Object.keys(freq).reduce((a, b) => freq[a] > freq[b] ? a : b, 'Unknown');
    });
    
    // Impute missing values
    return data.map(row => {
        const newRow = { ...row };
        DATA_SCHEMA.numerical.forEach(col => {
            if (newRow[col] === null || isNaN(newRow[col])) {
                newRow[col] = medians[col] || 0;
            }
        });
        DATA_SCHEMA.categorical.forEach(col => {
            if (!newRow[col] || newRow[col] === '') {
                newRow[col] = modes[col] || 'Unknown';
            }
        });
        return newRow;
    });
}

// One-hot encode categorical variables
function oneHotEncode(data) {
    const encoded = [];
    const featureNames = [];
    const categories = {};
    
    // Initialize categories
    DATA_SCHEMA.categorical.forEach(col => {
        const uniqueVals = [...new Set(data.map(row => row[col]).filter(val => val !== null && val !== undefined))];
        categories[col] = uniqueVals;
        uniqueVals.forEach(val => {
            featureNames.push(`${col}_${val}`);
        });
    });
    
    // Add numerical feature names
    DATA_SCHEMA.numerical.forEach(col => {
        featureNames.push(col);
    });
    
    // Encode each row
    data.forEach(row => {
        const rowEncoded = [];
        
        // One-hot encode categorical features
        DATA_SCHEMA.categorical.forEach(col => {
            const uniqueVals = categories[col];
            uniqueVals.forEach(val => {
                rowEncoded.push(row[col] === val ? 1 : 0);
            });
        });
        
        // Add numerical features
        DATA_SCHEMA.numerical.forEach(col => {
            rowEncoded.push(row[col] || 0);
        });
        
        encoded.push(rowEncoded);
    });
    
    return { features: encoded, featureNames };
}

// Standardize numerical features
function standardizeNumerical(features, featureNames) {
    const numericalIndices = [];
    
    // Find indices of numerical features
    DATA_SCHEMA.numerical.forEach(col => {
        const index = featureNames.indexOf(col);
        if (index !== -1) {
            numericalIndices.push(index);
        }
    });
    
    // Calculate mean and std for each numerical feature
    const means = [];
    const stds = [];
    
    numericalIndices.forEach(colIndex => {
        const values = features.map(row => row[colIndex]);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
        means.push(mean);
        stds.push(std);
    });
    
    // Standardize features
    return features.map(row => {
        const newRow = [...row];
        numericalIndices.forEach((colIndex, i) => {
            if (stds[i] !== 0) {
                newRow[colIndex] = (newRow[colIndex] - means[i]) / stds[i];
            }
        });
        return newRow;
    });
}

// Create the neural network model
function createModel() {
    const statusDiv = document.getElementById('modelStatus');
    const summaryDiv = document.getElementById('modelSummary');
    
    if (!trainData || !trainData.features) {
        alert('Please preprocess data first');
        return;
    }
    
    try {
        const inputShape = trainData.features.shape[1];
        
        // Create shallow neural network - single hidden layer
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
        summaryDiv.innerHTML = `<h4>Model Summary:</h4>`;
        
        let summaryText = 'Layer (type)                 Output shape              Param #\n';
        summaryText += '=================================================================\n';
        
        model.layers.forEach(layer => {
            const outputShape = layer.outputShape ? `[null, ${layer.outputShape[1]}]` : 'multiple';
            const params = layer.countParams();
            summaryText += `${layer.name.padEnd(20)} ${outputShape.padEnd(25)} ${params}\n`;
        });
        
        summaryText += `=================================================================\n`;
        summaryText += `Total params: ${model.countParams()}\n`;
        summaryText += `Trainable params: ${model.countParams()}\n`;
        summaryText += `Non-trainable params: 0\n`;
        
        summaryDiv.innerHTML += `<pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">${summaryText}</pre>`;
        
        statusDiv.innerHTML = '<div class="status status-success">Model created successfully!</div>';
        
    } catch (error) {
        console.error('Error creating model:', error);
        statusDiv.innerHTML = `<div class="status status-error">Error creating model: ${error.message}</div>`;
    }
}

// Train the model
async function trainModel() {
    const statusDiv = document.getElementById('trainingStatus');
    const trainBtn = document.getElementById('trainBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (!model) {
        alert('Please create model first');
        return;
    }
    
    if (!trainData || !trainData.features) {
        alert('Please preprocess data first');
        return;
    }
    
    // Get training parameters
    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batchSize').value);
    
    // Enable stop button, disable train button
    trainBtn.disabled = true;
    stopBtn.disabled = false;
    
    statusDiv.innerHTML = '<div class="status status-info">Training started...</div>';
    
    try {
        // Split data into train/validation (80/20)
        const splitIndex = Math.floor(trainData.features.shape[0] * 0.8);
        
        const trainFeatures = trainData.features.slice(0, splitIndex);
        const trainTargets = trainData.targets.slice(0, splitIndex);
        const valFeatures = trainData.features.slice(splitIndex);
        const valTargets = trainData.targets.slice(splitIndex);
        
        validationData = {
            features: valFeatures,
            targets: valTargets
        };
        
        // Set up callbacks for visualization
        const callbacks = tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Training' },
            ['loss', 'val_loss', 'acc', 'val_acc'],
            {
                callbacks: ['onEpochEnd', 'onBatchEnd'],
                height: 300
            }
        );
        
        // Train the model
        await model.fit(trainFeatures, trainTargets, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [valFeatures, valTargets],
            callbacks: callbacks,
            verbose: 1
        });
        
        trainBtn.disabled = false;
        stopBtn.disabled = true;
        statusDiv.innerHTML = '<div class="status status-success">Training completed!</div>';
        
    } catch (error) {
        console.error('Error training model:', error);
        statusDiv.innerHTML = `<div class="status status-error">Error training model: ${error.message}</div>`;
        trainBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// Stop training
function stopTraining() {
    // TensorFlow.js doesn't have direct stop training, but we can disable the button
    document.getElementById('trainBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('trainingStatus').innerHTML = '<div class="status status-info">Training stopped by user</div>';
}

// Evaluate the model
async function evaluateModel() {
    const metricsDiv = document.getElementById('metricsDisplay');
    const confusionDiv = document.getElementById('confusionMatrix');
    
    if (!model || !validationData) {
        alert('Please train the model first');
        return;
    }
    
    try {
        // Get predictions
        const predictions = model.predict(validationData.features);
        const probs = await predictions.data();
        predictions.dispose();
        
        const trueLabels = await validationData.targets.data();
        
        // Calculate metrics
        const metrics = calculateMetrics(trueLabels, probs, currentThreshold);
        
        // Display metrics
        metricsDiv.innerHTML = `
            <div class="metric-card">
                <div class="metric-value">${metrics.accuracy.toFixed(3)}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${metrics.precision.toFixed(3)}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${metrics.recall.toFixed(3)}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${metrics.f1.toFixed(3)}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${metrics.auc.toFixed(3)}</div>
                <div class="metric-label">AUC-ROC</div>
            </div>
        `;
        
        // Plot ROC curve
        plotROCCurve(trueLabels, probs);
        
        // Display confusion matrix
        displayConfusionMatrix(metrics.confusionMatrix, confusionDiv);
        
    } catch (error) {
        console.error('Error
