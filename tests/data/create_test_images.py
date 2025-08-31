"""
Script to create test images for the test suite.
Run this script to generate golden set test images.
"""

import os
from PIL import Image, ImageDraw
import numpy as np


def create_test_images():
    """Create test images for various test scenarios."""
    
    # Ensure data directory exists
    os.makedirs('tests/data', exist_ok=True)
    
    # 1. Simple test image with basic shapes (simulating person/car shapes)
    img = Image.new('RGB', (640, 480), color='skyblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a rectangle (simulating a car)
    draw.rectangle([100, 300, 250, 400], fill='red', outline='black', width=2)
    
    # Draw an oval (simulating a person)
    draw.ellipse([300, 200, 350, 350], fill='blue', outline='black', width=2)
    
    img.save('tests/data/test_image_with_objects.jpg', 'JPEG', quality=85)
    
    # 2. Empty image (no objects)
    empty_img = Image.new('RGB', (400, 300), color='white')
    empty_img.save('tests/data/empty_image.jpg', 'JPEG', quality=85)
    
    # 3. High resolution image for performance testing
    big_img = Image.new('RGB', (1920, 1080), color='lightgray')
    big_draw = ImageDraw.Draw(big_img)
    
    # Add some objects to the big image
    big_draw.rectangle([200, 600, 400, 800], fill='green', outline='black', width=3)
    big_draw.ellipse([800, 400, 900, 650], fill='yellow', outline='black', width=3)
    big_draw.rectangle([1200, 300, 1500, 500], fill='purple', outline='black', width=3)
    
    big_img.save('tests/data/high_res_test_image.jpg', 'JPEG', quality=90)
    
    # 4. 720p image for performance tests
    img_720p = Image.new('RGB', (1280, 720), color='lightblue')
    draw_720p = ImageDraw.Draw(img_720p)
    
    # Add multiple objects
    draw_720p.rectangle([100, 400, 300, 600], fill='red', outline='black', width=2)
    draw_720p.rectangle([500, 300, 700, 500], fill='blue', outline='black', width=2)
    draw_720p.ellipse([800, 200, 900, 400], fill='green', outline='black', width=2)
    
    img_720p.save('tests/data/test_720p.jpg', 'JPEG', quality=85)
    
    # 5. Small image
    small_img = Image.new('RGB', (64, 64), color='pink')
    small_draw = ImageDraw.Draw(small_img)
    small_draw.rectangle([10, 20, 50, 50], fill='black')
    small_img.save('tests/data/small_test.jpg', 'JPEG', quality=85)
    
    # 6. PNG test image
    png_img = Image.new('RGBA', (300, 300), color=(255, 255, 255, 255))
    png_draw = ImageDraw.Draw(png_img)
    png_draw.ellipse([50, 50, 250, 250], fill=(255, 0, 0, 128), outline='black', width=2)
    png_img.save('tests/data/test_image.png', 'PNG')
    
    print("Test images created successfully:")
    print("- tests/data/test_image_with_objects.jpg")
    print("- tests/data/empty_image.jpg") 
    print("- tests/data/high_res_test_image.jpg")
    print("- tests/data/test_720p.jpg")
    print("- tests/data/small_test.jpg")
    print("- tests/data/test_image.png")


if __name__ == "__main__":
    create_test_images()