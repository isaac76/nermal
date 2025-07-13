#ifndef DRAWINGWIDGET_H
#define DRAWINGWIDGET_H

#include <QWidget>
#include <QPainter>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPixmap>
#include <QPoint>
#include <QPen>
#include <vector>

class DrawingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DrawingWidget(QWidget *parent = nullptr);
    
    // Get the drawn image as a 28x28 normalized vector for neural network input
    std::vector<double> getImageAsVector() const;
    
    // Clear the drawing area
    void clearDrawing();
    
    // Set the drawing pen properties
    void setPenWidth(int width);
    void setPenColor(const QColor &color);

signals:
    // Emitted when the user finishes drawing (mouse release)
    void drawingCompleted();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    // Convert the current drawing to a 28x28 grayscale image
    QImage convertTo28x28() const;
    
    // Normalize pixel values to range [0.01, 0.99] like MNIST data
    std::vector<double> normalizePixelValues(const QImage &image) const;
    
    QPixmap m_canvas;          // The drawing canvas
    QPen m_pen;               // Drawing pen
    QPoint m_lastPoint;       // Last mouse position
    bool m_drawing;           // Whether currently drawing
    
    static const int CANVAS_SIZE = 280;  // Size of the drawing area (matches UI)
};

#endif // DRAWINGWIDGET_H
