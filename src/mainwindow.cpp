#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QApplication>
#include <QCoreApplication>
#include <QMessageBox>
#include <QFileInfo>
#include <QDir>
#include <QTextStream>
#include <QDebug>
#include <QThread>
#include <QStatusBar>
#include <QPushButton>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QImage>
#include <QGraphicsView>
#include <QRandomGenerator>
#include <QTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , neuralNetwork(nullptr)
    , progressBar(nullptr)
    , statusLabel(nullptr)
    , imageScene(nullptr)
    , imageItem(nullptr)
    , testImageScene(nullptr)
    , testImageItem(nullptr)
{
    ui->setupUi(this);
    
    // Initialize neural network (784 inputs for 28x28 MNIST, 100 hidden, 10 outputs for digits 0-9)
    neuralNetwork = new NeuralNetwork(784, 100, 10, 0.1);
    
    // Setup additional UI elements
    setupUI();
}

MainWindow::~MainWindow()
{
    delete ui;
    delete neuralNetwork;
    delete imageScene;
    delete testImageScene;
}


void MainWindow::setupUI()
{
    // Create progress bar for training feedback
    progressBar = new QProgressBar(this);
    progressBar->setVisible(false);
    progressBar->setRange(0, 100);
    
    // Use the statusLabel from the UI file instead of creating our own
    statusLabel = ui->statusLabel;
    statusLabel->setText("Neural Network Ready");
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setStyleSheet("QLabel { color: blue; font-weight: bold; padding: 5px; }");
    
    // Find the main vertical layout and add the progress bar
    QVBoxLayout* mainLayout = ui->centralwidget->findChild<QVBoxLayout*>("verticalLayout");
    if (mainLayout) {
        // Insert progress bar into the layout
        mainLayout->insertWidget(1, progressBar);
    } else {
        // Fallback: use status bar if layout not found
        statusBar()->addPermanentWidget(progressBar);
    }
    
    // Setup graphics view for image display
    imageScene = new QGraphicsScene(this);
    ui->graphicsView->setScene(imageScene);
    
    // Configure graphics view for MNIST image display
    ui->graphicsView->setStyleSheet("QGraphicsView { border: 2px solid gray; background-color: white; }");
    
    // Hide scroll bars
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    
    // Create a placeholder image item
    imageItem = new QGraphicsPixmapItem();
    imageScene->addItem(imageItem);
    
    // Set initial scene rect - will be updated when images are displayed
    imageScene->setSceneRect(0, 0, 280, 280);
    
    // Setup test image view
    testImageScene = new QGraphicsScene(this);
    ui->testImageView->setScene(testImageScene);
    
    // Configure test image view for MNIST image display
    ui->testImageView->setStyleSheet("QGraphicsView { border: 2px solid blue; background-color: white; }");
    
    // Hide scroll bars for test image view
    ui->testImageView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->testImageView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    
    // Create a placeholder test image item
    testImageItem = new QGraphicsPixmapItem();
    testImageScene->addItem(testImageItem);
    
    // Set initial scene rect for test image
    testImageScene->setSceneRect(0, 0, 280, 280);
    
    // Set window title
    setWindowTitle("Nermal Neural Network - MNIST Digit Recognition");
}

void MainWindow::loadMNISTData(const QString& filename, std::vector<std::vector<double>>& inputs, 
                               std::vector<std::vector<double>>& targets)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "Error", "Cannot open file: " + filename);
        return;
    }
    
    QTextStream in(&file);
    QString line;
    
    // Skip header if present
    if (!in.atEnd()) {
        line = in.readLine();
        // Check if this looks like a header (non-numeric first character)
        if (!line.isEmpty() && !line.at(0).isDigit()) {
            // Skip this line as it's likely a header
        } else {
            // This is data, so reset to beginning
            in.seek(0);
        }
    }
    
    int count = 0;
    while (!in.atEnd() && count < 10000) { // Limit to 10k samples for initial training
        line = in.readLine();
        if (line.isEmpty()) continue;
        
        QStringList values = line.split(',');
        if (values.size() < 785) continue; // 1 label + 784 pixels
        
        // First value is the label (digit 0-9)
        int label = values[0].toInt();
        
        // Create target vector (one-hot encoding)
        std::vector<double> target(10, 0.01); // Small value instead of 0
        target[label] = 0.99; // High value instead of 1
        targets.push_back(target);
        
        // Remaining values are pixel intensities
        std::vector<double> input;
        for (int i = 1; i < values.size(); ++i) {
            double pixel = values[i].toDouble();
            // Normalize pixel value to range [0.01, 0.99]
            input.push_back((pixel / 255.0) * 0.98 + 0.01);
        }
        inputs.push_back(input);
        
        count++;
    }
    
    qDebug() << "Loaded" << count << "training samples from" << filename;
}

void MainWindow::displayMNISTImage(const std::vector<double>& imageData, int label)
{
    if (imageData.size() != 784) {
        qDebug() << "Invalid image data size:" << imageData.size() << "(expected 784)";
        return;
    }
    
    // Create a 28x28 QImage from the image data
    QImage image(28, 28, QImage::Format_Grayscale8);
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int index = y * 28 + x;
            
            // Convert normalized value [0.01, 0.99] back to grayscale [0, 255]
            double normalizedValue = imageData[index];
            // Reverse the normalization: (value - 0.01) / 0.98 * 255
            int grayValue = static_cast<int>((normalizedValue - 0.01) / 0.98 * 255.0);
            grayValue = qBound(0, grayValue, 255);
            
            image.setPixel(x, y, grayValue);
        }
    }
    
    // Convert to pixmap and display in graphics view
    QPixmap pixmap = QPixmap::fromImage(image);
    
    // Scale up the image for better visibility (10x larger)
    pixmap = pixmap.scaled(280, 280, Qt::KeepAspectRatio, Qt::FastTransformation);
    
    imageItem->setPixmap(pixmap);
    
    // Position the image at the top-left corner of the scene
    imageItem->setPos(0, 0);
    
    // Update the scene rect to match the pixmap size
    imageScene->setSceneRect(0, 0, pixmap.width(), pixmap.height());
    
    // Update status to show the label if provided
    if (label >= 0) {
        statusLabel->setText(QString("Displaying digit: %1").arg(label));
    } else {
        statusLabel->setText("Displaying MNIST image");
    }
    
    // Fit the image in the view properly
    ui->graphicsView->fitInView(imageScene->sceneRect(), Qt::KeepAspectRatio);
}

void MainWindow::displayTestImage(const std::vector<double>& imageData)
{
    if (imageData.size() != 784) {
        qDebug() << "Invalid test image data size:" << imageData.size() << "(expected 784)";
        return;
    }
    
    // Create a 28x28 QImage from the image data
    QImage image(28, 28, QImage::Format_Grayscale8);
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            int index = y * 28 + x;
            
            // Convert normalized value [0.01, 0.99] back to grayscale [0, 255]
            double normalizedValue = imageData[index];
            // Reverse the normalization: (value - 0.01) / 0.98 * 255
            int grayValue = static_cast<int>((normalizedValue - 0.01) / 0.98 * 255.0);
            grayValue = qBound(0, grayValue, 255);
            
            image.setPixel(x, y, grayValue);
        }
    }
    
    // Convert to pixmap and display in test image view
    QPixmap pixmap = QPixmap::fromImage(image);
    
    // Scale up the image for better visibility (10x larger)
    pixmap = pixmap.scaled(280, 280, Qt::KeepAspectRatio, Qt::FastTransformation);
    
    testImageItem->setPixmap(pixmap);
    
    // Position the image at the top-left corner of the scene
    testImageItem->setPos(0, 0);
    
    // Update the scene rect to match the pixmap size
    testImageScene->setSceneRect(0, 0, pixmap.width(), pixmap.height());
    
    // Fit the image in the test view properly
    ui->testImageView->fitInView(testImageScene->sceneRect(), Qt::KeepAspectRatio);
}

int MainWindow::predictDigit(const std::vector<double>& imageData)
{
    if (!neuralNetwork || imageData.size() != 784) {
        return -1;
    }
    
    // Get prediction from neural network
    std::vector<double> outputs = neuralNetwork->query(imageData);
    
    // Find the index with the highest output value
    int predictedDigit = 0;
    double maxOutput = outputs[0];
    
    for (int i = 1; i < static_cast<int>(outputs.size()); ++i) {
        if (outputs[i] > maxOutput) {
            maxOutput = outputs[i];
            predictedDigit = i;
        }
    }
    
    return predictedDigit;
}

void MainWindow::on_train_clicked()
{
    statusLabel->setText("Starting training...");
    progressBar->setVisible(true);
    progressBar->setValue(0);
    
    // Disable train button during training
    ui->train->setEnabled(false);
    
    QApplication::processEvents(); // Update UI
    
    // Load training data
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
    
    // Look for MNIST training file
    QString trainingFile;
    
    // Try different possible locations
    QStringList possiblePaths = {
        QDir::currentPath() + "/../training/mnist_train.csv",           // From build directory
        QDir::currentPath() + "/training/mnist_train.csv",              // From project root
        QDir::currentPath() + "/../../training/mnist_train.csv",        // From nested build dir
        QCoreApplication::applicationDirPath() + "/../training/mnist_train.csv",  // Relative to executable
        QCoreApplication::applicationDirPath() + "/../../training/mnist_train.csv", // From deep build
        "training/mnist_train.csv"                                      // Relative path
    };
    
    // Find the first existing file
    for (const QString& path : possiblePaths) {
        if (QFileInfo::exists(path)) {
            trainingFile = path;
            qDebug() << "Found training file at:" << trainingFile;
            break;
        }
    }
    
    // If still not found, show what we tried
    if (trainingFile.isEmpty()) {
        QString searchedPaths;
        for (const QString& path : possiblePaths) {
            searchedPaths += path + "\n";
        }
        QMessageBox::critical(this, "Training File Not Found", 
                             QString("Could not find mnist_train.csv in any of these locations:\n\n%1\n"
                                     "Current working directory: %2\n"
                                     "Application directory: %3")
                             .arg(searchedPaths)
                             .arg(QDir::currentPath())
                             .arg(QCoreApplication::applicationDirPath()));
        progressBar->setVisible(false);
        ui->train->setEnabled(true);
        statusLabel->setText("Training failed - file not found");
        return;
    }
    
    loadMNISTData(trainingFile, inputs, targets);
    
    if (inputs.empty()) {
        QMessageBox::critical(this, "Error", "Failed to load training data from: " + trainingFile);
        progressBar->setVisible(false);
        ui->train->setEnabled(true);
        statusLabel->setText("Training failed - no data loaded");
        return;
    }
    
    // Display the first training image as a sample
    if (!inputs.empty()) {
        // Find the label for the first image
        int firstLabel = -1;
        if (!targets.empty() && targets[0].size() == 10) {
            for (int i = 0; i < 10; ++i) {
                if (targets[0][i] > 0.5) { // Find the highest value (should be 0.99)
                    firstLabel = i;
                    break;
                }
            }
        }
        displayMNISTImage(inputs[0], firstLabel);
    }
    
    // Training parameters
    const int epochs = 5;
    const int totalSamples = inputs.size();
    
    statusLabel->setText(QString("Training on %1 samples for %2 epochs...").arg(totalSamples).arg(epochs));
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        statusLabel->setText(QString("Training epoch %1/%2...").arg(epoch + 1).arg(epochs));
        
        for (int i = 0; i < totalSamples; ++i) {
            neuralNetwork->train(inputs[i], targets[i]);
            
            // Update progress
            int progress = ((epoch * totalSamples + i + 1) * 100) / (epochs * totalSamples);
            progressBar->setValue(progress);
            
            // Update UI periodically and show different training images
            if (i % 1000 == 0) {
                QApplication::processEvents();
                
                // Display a different training image every 1000 samples
                if (i > 0) {
                    // Find the label for this image
                    int currentLabel = -1;
                    if (targets[i].size() == 10) {
                        for (int j = 0; j < 10; ++j) {
                            if (targets[i][j] > 0.5) {
                                currentLabel = j;
                                break;
                            }
                        }
                    }
                    displayMNISTImage(inputs[i], currentLabel);
                }
            }
        }
        
        qDebug() << "Completed epoch" << (epoch + 1) << "of" << epochs;
    }
    
    // Training complete
    progressBar->setVisible(false);
    statusLabel->setText("Training completed successfully!");
    ui->train->setEnabled(true);
    
    QMessageBox::information(this, "Training Complete", 
                           QString("Neural network training completed!\n\n"
                                   "Trained on %1 samples over %2 epochs.\n"
                                   "The network is now ready for digit recognition.")
                           .arg(totalSamples).arg(epochs));
}


void MainWindow::on_query_clicked()
{
    // Check if neural network has been trained
    if (!neuralNetwork) {
        QMessageBox::warning(this, "Not Ready", "Please train the neural network first!");
        return;
    }
    
    statusLabel->setText("Loading test data...");
    QApplication::processEvents();
    
    // Load test data
    std::vector<std::vector<double>> testInputs;
    std::vector<std::vector<double>> testTargets;
    
    // Look for MNIST test file
    QString testFile;
    
    // Try different possible locations for test file
    QStringList possiblePaths = {
        QDir::currentPath() + "/../mnist_test.csv",                     // From build directory to project root
        QDir::currentPath() + "/mnist_test.csv",                       // From project root
        QDir::currentPath() + "/../../mnist_test.csv",                 // From nested build dir
        QCoreApplication::applicationDirPath() + "/../mnist_test.csv", // Relative to executable
        QCoreApplication::applicationDirPath() + "/../../mnist_test.csv", // From deep build
        QDir::currentPath() + "/../training/mnist_test.csv",           // From build directory to training folder
        QDir::currentPath() + "/training/mnist_test.csv",              // From project root to training
        QDir::currentPath() + "/../../training/mnist_test.csv",        // From nested build dir to training
        QCoreApplication::applicationDirPath() + "/../training/mnist_test.csv",  // Relative to executable to training
        QCoreApplication::applicationDirPath() + "/../../training/mnist_test.csv", // From deep build to training
        "mnist_test.csv"                                               // Current directory
    };
    
    // Find the first existing file
    for (const QString& path : possiblePaths) {
        if (QFileInfo::exists(path)) {
            testFile = path;
            qDebug() << "Found test file at:" << testFile;
            break;
        }
    }
    
    // If still not found, show what we tried
    if (testFile.isEmpty()) {
        QString searchedPaths;
        for (const QString& path : possiblePaths) {
            searchedPaths += path + "\n";
        }
        QMessageBox::critical(this, "Test File Not Found", 
                             QString("Could not find mnist_test.csv in any of these locations:\n\n%1\n"
                                     "Current working directory: %2\n"
                                     "Application directory: %3")
                             .arg(searchedPaths)
                             .arg(QDir::currentPath())
                             .arg(QCoreApplication::applicationDirPath()));
        statusLabel->setText("Query failed - test file not found");
        return;
    }
    
    // Load a limited amount of test data (1000 samples for quick random selection)
    loadMNISTData(testFile, testInputs, testTargets);
    
    if (testInputs.empty()) {
        QMessageBox::critical(this, "Error", "Failed to load test data from: " + testFile);
        statusLabel->setText("Query failed - no test data loaded");
        return;
    }
    
    // Randomly select a test image
    int randomIndex = QRandomGenerator::global()->bounded(static_cast<int>(testInputs.size()));
    
    // Get the actual label from the target vector
    int actualLabel = -1;
    if (randomIndex < static_cast<int>(testTargets.size()) && testTargets[randomIndex].size() == 10) {
        for (int i = 0; i < 10; ++i) {
            if (testTargets[randomIndex][i] > 0.5) {
                actualLabel = i;
                break;
            }
        }
    }
    
    // Display the test image
    displayTestImage(testInputs[randomIndex]);
    
    // Get prediction from neural network
    int prediction = predictDigit(testInputs[randomIndex]);
    
    // Display the prediction in the edit box
    ui->predictionEdit->setText(QString::number(prediction));
    
    // Update status with both actual and predicted values
    if (actualLabel >= 0) {
        statusLabel->setText(QString("Actual: %1, Predicted: %2 %3")
                           .arg(actualLabel)
                           .arg(prediction)
                           .arg(prediction == actualLabel ? "✓" : "✗"));
    } else {
        statusLabel->setText(QString("Predicted: %1").arg(prediction));
    }
    
    qDebug() << "Test sample" << randomIndex << "- Actual:" << actualLabel << "Predicted:" << prediction;
}
