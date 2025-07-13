#include "drawingwidget.h"
#include <QDebug>
#include <QPainterPath>
#include <cmath>

DrawingWidget::DrawingWidget(QWidget *parent)
    : QWidget(parent)
    , m_drawing(false)
{
    // Set up the widget
    setFixedSize(CANVAS_SIZE, CANVAS_SIZE);
    setAttribute(Qt::WA_StaticContents);
    
    // Initialize the canvas
    m_canvas = QPixmap(CANVAS_SIZE, CANVAS_SIZE);
    m_canvas.fill(Qt::white);  // Start with white background (like paper)
    
    // Set up the drawing pen
    m_pen.setColor(Qt::black);
    m_pen.setWidth(8);  // Thick pen for digit drawing
    m_pen.setStyle(Qt::SolidLine);
    m_pen.setCapStyle(Qt::RoundCap);
    m_pen.setJoinStyle(Qt::RoundJoin);
    
    // Set background and border styling
    setStyleSheet("DrawingWidget { border: 2px solid #666; background-color: white; }");
}

void DrawingWidget::paintEvent(QPaintEvent *event)
{
    QPainter painter(this);
    QRect dirtyRect = event->rect();
    painter.drawPixmap(dirtyRect, m_canvas, dirtyRect);
}

void DrawingWidget::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        m_lastPoint = event->pos();
        m_drawing = true;
    }
}

void DrawingWidget::mouseMoveEvent(QMouseEvent *event)
{
    if ((event->buttons() & Qt::LeftButton) && m_drawing) {
        QPainter painter(&m_canvas);
        painter.setPen(m_pen);
        painter.drawLine(m_lastPoint, event->pos());
        
        // Update only the drawn area for efficiency
        int rad = (m_pen.width() / 2) + 2;
        update(QRect(m_lastPoint, event->pos()).normalized()
               .adjusted(-rad, -rad, +rad, +rad));
        
        m_lastPoint = event->pos();
    }
}

void DrawingWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && m_drawing) {
        m_drawing = false;
        emit drawingCompleted();  // Signal that drawing is finished
    }
}

void DrawingWidget::clearDrawing()
{
    m_canvas.fill(Qt::white);
    update();  // Trigger repaint
}

void DrawingWidget::setPenWidth(int width)
{
    m_pen.setWidth(width);
}

void DrawingWidget::setPenColor(const QColor &color)
{
    m_pen.setColor(color);
}

QImage DrawingWidget::convertTo28x28() const
{
    // Convert pixmap to image
    QImage originalImage = m_canvas.toImage();
    
    // Convert to grayscale if not already
    if (originalImage.format() != QImage::Format_Grayscale8) {
        originalImage = originalImage.convertToFormat(QImage::Format_Grayscale8);
    }
    
    // Find the bounding box of the drawing (non-white pixels)
    QRect boundingBox;
    bool foundContent = false;
    
    for (int y = 0; y < originalImage.height() && !foundContent; ++y) {
        for (int x = 0; x < originalImage.width(); ++x) {
            if (qGray(originalImage.pixel(x, y)) < 250) {  // Not white (allowing for slight variations)
                boundingBox.setTop(y);
                foundContent = true;
                break;
            }
        }
    }
    
    if (!foundContent) {
        // No drawing found, return empty 28x28 image
        return QImage(28, 28, QImage::Format_Grayscale8);
    }
    
    // Find bottom, left, and right bounds
    for (int y = originalImage.height() - 1; y >= 0; --y) {
        for (int x = 0; x < originalImage.width(); ++x) {
            if (qGray(originalImage.pixel(x, y)) < 250) {
                boundingBox.setBottom(y);
                goto found_bottom;
            }
        }
    }
    found_bottom:
    
    for (int x = 0; x < originalImage.width(); ++x) {
        for (int y = 0; y < originalImage.height(); ++y) {
            if (qGray(originalImage.pixel(x, y)) < 250) {
                boundingBox.setLeft(x);
                goto found_left;
            }
        }
    }
    found_left:
    
    for (int x = originalImage.width() - 1; x >= 0; --x) {
        for (int y = 0; y < originalImage.height(); ++y) {
            if (qGray(originalImage.pixel(x, y)) < 250) {
                boundingBox.setRight(x);
                goto found_right;
            }
        }
    }
    found_right:
    
    // Add some padding around the bounding box (like MNIST preprocessing)
    int padding = std::max(boundingBox.width(), boundingBox.height()) * 0.1;
    boundingBox.adjust(-padding, -padding, padding, padding);
    
    // Ensure bounding box is within image bounds
    boundingBox = boundingBox.intersected(originalImage.rect());
    
    // Crop to bounding box
    QImage croppedImage = originalImage.copy(boundingBox);
    
    // Scale to fit in 20x20 (like MNIST center area), then center in 28x28
    QImage scaledImage = croppedImage.scaled(20, 20, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    
    // Create 28x28 image and center the scaled image
    QImage result(28, 28, QImage::Format_Grayscale8);
    result.fill(255);  // White background
    
    // Calculate centering offset
    int offsetX = (28 - scaledImage.width()) / 2;
    int offsetY = (28 - scaledImage.height()) / 2;
    
    // Copy scaled image to center of result
    QPainter painter(&result);
    painter.drawImage(offsetX, offsetY, scaledImage);
    
    return result;
}

std::vector<double> DrawingWidget::normalizePixelValues(const QImage &image) const
{
    std::vector<double> normalizedValues;
    normalizedValues.reserve(784);  // 28 * 28
    
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            // Get grayscale value (0-255)
            int grayValue = qGray(image.pixel(x, y));
            
            // Invert (MNIST has white digits on black background, we have black on white)
            grayValue = 255 - grayValue;
            
            // Normalize to [0.01, 0.99] range like MNIST training data
            double normalized = (static_cast<double>(grayValue) / 255.0) * 0.98 + 0.01;
            
            normalizedValues.push_back(normalized);
        }
    }
    
    return normalizedValues;
}

std::vector<double> DrawingWidget::getImageAsVector() const
{
    QImage image28x28 = convertTo28x28();
    return normalizePixelValues(image28x28);
}
