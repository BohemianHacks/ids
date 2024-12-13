## ids

JavaScript implementation for a Network Intrusion Detection System (NIDS) using Node.js.

This code you provided implements a Network Intrusion Detection System (NIDS) using Node.js and TensorFlow.js. Here's a breakdown of the key components:

**System Architecture:**

* **NetworkIntrusionDetectionSystem:** This is the main class that coordinates all the other components. It handles initialization, model loading/training, starting packet capture, and intrusion detection.
* **NetworkTrafficAnalyzer:** This class is responsible for extracting relevant features from captured network packets. It preprocesses the raw data into a format suitable for the machine learning model.
* **ModelTrainer:** This class handles creating, training, and updating the machine learning model used for intrusion detection. It loads training data, defines the model architecture, and trains it to identify potential threats.
* **LoggerService:**  This class handles logging messages with different severities (info, error, alert). It allows logging to the console and files for later analysis.
* **DataLoader:** This class manages loading training data from various sources like JSON files. It can also generate synthetic data if real data is unavailable.

**Key Features:**

* **Machine learning-based intrusion detection:** The system uses a machine learning model to analyze network traffic and identify potential attacks.
* **Real-time packet capture and analysis:** Packets are captured from the network interface in real-time and analyzed for suspicious activity.
* **Dynamic model training:** The model can be periodically updated with new data to improve its accuracy over time.

**Code walkthrough:**

1. The provided code snippets define each class mentioned above. 
2. `NetworkIntrusionDetectionSystem` uses libraries like `pcap` for packet capture and `tfjs-node` for TensorFlow functionality.
3. `NetworkTrafficAnalyzer` extracts features like source/destination IPs, protocols, packet lengths, etc. from captured packets.
4. `ModelTrainer` defines a simple sequential neural network architecture with hidden layers and uses `tf.train.adam` optimizer for training.
5. `LoggerService` uses the `winston` library to manage different log levels and destinations (console and file).
6. `loadTrainingData` attempts to load training data from JSON files and converts them to tensors suitable for the model. It falls back to generating synthetic data if real data is unavailable.
7. Finally, `main.js` demonstrates how to instantiate the `NetworkIntrusionDetectionSystem` and starts capturing packets. It also sets up a graceful shutdown handler for the system.

Overall, this code provides a good foundation for building a basic NIDS using Node.js and machine learning. It demonstrates key functionalities like packet capture, feature extraction, model training, and intrusion detection with logging.

Here are some additional points to consider:

* The model architecture is a basic example. You can explore more complex architectures for improved accuracy.
* The code doesn't explicitly handle integrating with threat intelligence feeds for real-time updates.
* Performance optimization techniques can be implemented for larger network traffic volumes.

```javascript
const pcap = require('pcap');
const tf = require('@tensorflow/tfjs-node');
const { NetworkTrafficAnalyzer } = require('./networkTrafficAnalyzer');
const { ModelTrainer } = require('./modelTrainer');
const { LoggerService } = require('./loggerService');
const config = require('./config');

class NetworkIntrusionDetectionSystem {
    constructor() {
        this.logger = new LoggerService();
        this.networkAnalyzer = new NetworkTrafficAnalyzer();
        this.modelTrainer = new ModelTrainer();
        this.model = null;
        this.isRunning = false;
    }

    async initialize() {
        try {
            // Load or train the model
            this.model = await this.loadOrTrainModel();
            this.logger.info('Intrusion Detection System initialized successfully');
        } catch (error) {
            this.logger.error('Initialization failed', error);
            throw error;
        }
    }

    async loadOrTrainModel() {
        try {
            // Try to load existing model
            const modelPath = config.MODEL_SAVE_PATH;
            let model;

            try {
                model = await tf.loadLayersModel(`file://${modelPath}`);
                this.logger.info('Existing model loaded successfully');
                return model;
            } catch (loadError) {
                this.logger.warn('No existing model found. Training new model...');
                model = await this.modelTrainer.trainModel();
                
                // Save the model
                await model.save(`file://${modelPath}`);
                this.logger.info('New model trained and saved');
                return model;
            }
        } catch (error) {
            this.logger.error('Model loading/training failed', error);
            throw error;
        }
    }

    startCapture(interfaceName = config.DEFAULT_INTERFACE) {
        if (this.isRunning) {
            this.logger.warn('Capture is already running');
            return;
        }

        try {
            const pcapSession = pcap.createSession(interfaceName, config.CAPTURE_FILTER);
            
            this.isRunning = true;
            this.logger.info(`Starting packet capture on interface: ${interfaceName}`);

            pcapSession.on('packet', async (rawPacket) => {
                try {
                    // Parse and preprocess packet
                    const packetFeatures = this.networkAnalyzer.extractFeatures(rawPacket);
                    
                    // Perform intrusion detection
                    const predictionTensor = tf.tidy(() => {
                        const inputTensor = tf.tensor2d([packetFeatures]);
                        return this.model.predict(inputTensor);
                    });

                    const prediction = await predictionTensor.data();
                    const threatScore = prediction[0];

                    // Log potential threats
                    if (threatScore > config.THREAT_THRESHOLD) {
                        this.logger.alert(`Potential Intrusion Detected! Threat Score: ${threatScore}`, {
                            packetDetails: packetFeatures,
                            threatScore
                        });
                    }

                    predictionTensor.dispose();
                } catch (processingError) {
                    this.logger.error('Packet processing error', processingError);
                }
            });

            // Handle capture termination
            pcapSession.on('error', (err) => {
                this.logger.error('Capture session error', err);
                this.stopCapture();
            });

        } catch (captureError) {
            this.logger.error('Failed to start packet capture', captureError);
            this.isRunning = false;
        }
    }

    stopCapture() {
        this.isRunning = false;
        this.logger.info('Packet capture stopped');
    }

    // Graceful shutdown method
    async shutdown() {
        this.stopCapture();
        await this.modelTrainer.updateModel();
    }
}

module.exports = { NetworkIntrusionDetectionSystem };

```

```javascript
const pcap = require('pcap');
const ip = require('ip');

class NetworkTrafficAnalyzer {
    extractFeatures(rawPacket) {
        try {
            const packet = pcap.decode.packet(rawPacket);
            
            // Extract network layer details
            const networkLayer = packet.payload.payload;
            
            // Basic feature extraction
            const features = [
                // Source IP (normalized)
                this.normalizeIP(networkLayer.saddr),
                // Destination IP (normalized)
                this.normalizeIP(networkLayer.daddr),
                // Protocol
                this.getProtocolCode(networkLayer.protocol),
                // Packet length
                rawPacket.length / 1500,  // Normalize by typical MTU
                // Time-based features
                this.getCurrentTimeFeature(),
                // Port information (if available)
                this.extractPortFeatures(networkLayer)
            ];

            return features;
        } catch (error) {
            console.error('Feature extraction error', error);
            return new Array(10).fill(0);  // Return default features on error
        }
    }

    normalizeIP(ipAddress) {
        try {
            const numericIP = ip.toLong(ipAddress);
            // Normalize IP to range [0, 1]
            return (numericIP % 4294967296) / 4294967296;
        } catch {
            return 0;
        }
    }

    getProtocolCode(protocol) {
        const protocolMap = {
            1: 0.1,   // ICMP
            6: 0.5,   // TCP
            17: 0.9   // UDP
        };
        return protocolMap[protocol] || 0;
    }

    getCurrentTimeFeature() {
        // Normalize current time of day
        const now = new Date();
        return now.getHours() / 24;
    }

    extractPortFeatures(networkLayer) {
        try {
            // Attempt to extract TCP/UDP port information
            const transportLayer = networkLayer.payload;
            const sourcePort = transportLayer.sport / 65535;
            const destPort = transportLayer.dport / 65535;
            
            return [sourcePort, destPort];
        } catch {
            return [0, 0];
        }
    }
}

module.exports = { NetworkTrafficAnalyzer };

```

```javascript
const tf = require('@tensorflow/tfjs-node');
const { loadTrainingData } = require('./dataLoader');
const config = require('./config');

class ModelTrainer {
    constructor() {
        this.model = null;
    }

    createModel() {
        const model = tf.sequential();

        // Input layer
        model.add(tf.layers.dense({
            inputShape: [config.FEATURE_COUNT],
            units: 64,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
        }));

        // Hidden layers
        model.add(tf.layers.dropout({ rate: 0.2 }));
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
        }));

        model.add(tf.layers.dropout({ rate: 0.2 }));

        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));

        return model;
    }

    async trainModel() {
        // Load training data
        const { xs, ys } = await loadTrainingData();

        // Create model
        const model = this.createModel();

        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        // Train model with early stopping and validation split
        await model.fit(xs, ys, {
            epochs: 50,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`);
                }
            }
        });

        return model;
    }

    async updateModel() {
        // Periodically retrain or fine-tune the model with new data
        // This method can be expanded to implement continual learning
        if (this.shouldRetrain()) {
            this.model = await this.trainModel();
            await this.saveModel();
        }
    }

    shouldRetrain() {
        // Implement logic to determine if retraining is necessary
        // Could be based on time, performance degradation, new data availability
        return Math.random() < 0.1;  // 10% chance of retraining for example
    }

    async saveModel() {
        await this.model.save(`file://${config.MODEL_SAVE_PATH}`);
    }
}

module.exports = { ModelTrainer };

```

```javascript
module.exports = {
    // Network Capture Configuration
    DEFAULT_INTERFACE: 'eth0',
    CAPTURE_FILTER: 'ip and tcp',
    
    // Model Configuration
    FEATURE_COUNT: 10,
    MODEL_SAVE_PATH: './models/intrusion_detection_model',
    
    // Detection Thresholds
    THREAT_THRESHOLD: 0.7,
    
    // Logging Configuration
    LOG_LEVEL: 'info',
    LOG_FILE: './logs/ids.log',
    
    // External Data Sources
    THREAT_INTELLIGENCE_SOURCES: [
        'https://example.com/threat-feeds',
        'https://another-threat-source.com/feeds'
    ],
    
    // Performance Monitoring
    PERFORMANCE_MONITORING: {
        enabled: true,
        interval: 300000, // 5 minutes
        metricsEndpoint: 'https://metrics.example.com/report'
    }
};

```

```javascript
const winston = require('winston');
const config = require('./config');

class LoggerService {
    constructor() {
        this.logger = winston.createLogger({
            level: config.LOG_LEVEL,
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.errors({ stack: true }),
                winston.format.splat(),
                winston.format.json()
            ),
            defaultMeta: { service: 'network-ids' },
            transports: [
                // Console transport
                new winston.transports.Console({
                    format: winston.format.simple()
                }),
                // File transport for persistent logging
                new winston.transports.File({
                    filename: config.LOG_FILE,
                    maxsize: 5242880, // 5MB
                    maxFiles: 5
                })
            ]
        });
    }

    info(message, meta = {}) {
        this.logger.info(message, meta);
    }

    error(message, error) {
        this.logger.error(message, { error: error.toString() });
    }

    alert(message, details) {
        // High-priority logging for potential threats
        this.logger.warn(message, {
            ...details,
            alertType: 'POTENTIAL_INTRUSION'
        });

        // Optional: Trigger external alert mechanisms
        this.triggerAlertMechanism(message, details);
    }

    triggerAlertMechanism(message, details) {
        // Placeholder for external alert integration
        // Could send emails, SMS, trigger webhooks, etc.
        console.log('EXTERNAL ALERT TRIGGERED:', message);
    }
}

module.exports = { LoggerService };

```

```javascript
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

async function loadTrainingData(dataPath = './training_data') {
    try {
        // Load multiple datasets
        const datasets = fs.readdirSync(dataPath)
            .filter(file => file.endsWith('.json'))
            .map(file => JSON.parse(fs.readFileSync(path.join(dataPath, file), 'utf8')));

        // Combine and preprocess datasets
        const combinedData = datasets.reduce((acc, dataset) => {
            acc.features.push(...dataset.features);
            acc.labels.push(...dataset.labels);
            return acc;
        }, { features: [], labels: [] });

        // Convert to tensors
        const xs = tf.tensor2d(combinedData.features);
        const ys = tf.tensor2d(combinedData.labels, [combinedData.labels.length, 1]);

        return { xs, ys };
    } catch (error) {
        console.error('Data loading error:', error);
        
        // Fallback to synthetic data if real data cannot be loaded
        return generateSyntheticData();
    }
}

function generateSyntheticData() {
    // Generate synthetic training data for initial model training
    const features = Array.from({ length: 1000 }, () => 
        Array.from({ length: 10 }, () => Math.random())
    );
    const labels = features.map(() => [Math.random() > 0.5 ? 1 : 0]);

    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels);

    return { xs, ys };
}

module.exports = { loadTrainingData };

```

```javascript
const { NetworkIntrusionDetectionSystem } = require('./networkIntrusionDetection');
const { LoggerService } = require('./loggerService');

async function main() {
    const logger = new LoggerService();
    const ids = new NetworkIntrusionDetectionSystem();

    try {
        // Initialize the system
        await ids.initialize();

        // Start packet capture
        ids.startCapture();

        // Handle graceful shutdown
        process.on('SIGINT', async () => {
            logger.info('Shutting down Network IDS...');
            await ids.shutdown();
            process.exit(0);
        });

    } catch (error) {
        logger.error('Failed to start Network Intrusion Detection System', error);
        process.exit(1);
    }
}

// Run the main application
main();

```
