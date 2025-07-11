#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProgressBar>
#include <QLabel>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include "neuralnetwork.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_train_clicked();
    void on_query_clicked();

private:
    Ui::MainWindow *ui;
    NeuralNetwork* neuralNetwork;
    QProgressBar* progressBar;
    QLabel* statusLabel;
    QGraphicsScene* imageScene;
    QGraphicsPixmapItem* imageItem;
    QGraphicsScene* testImageScene;
    QGraphicsPixmapItem* testImageItem;
    
    // Helper methods
    void setupUI();
    void loadMNISTData(const QString& filename, std::vector<std::vector<double>>& inputs, 
                       std::vector<std::vector<double>>& targets);
    void displayMNISTImage(const std::vector<double>& imageData, int label = -1);
    void displayTestImage(const std::vector<double>& imageData);
    int predictDigit(const std::vector<double>& imageData);
};

#endif // MAINWINDOW_H
