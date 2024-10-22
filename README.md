🚧 Automating Defect Detection in Cast Products Using AI 🚧

  I’m excited to share our recent project on automating defect detection in casting products with advanced machine learning models, a significant leap forward in enhancing manufacturing    quality control! 📊
  
  In collaboration with my teammates, Sameer Kandathinkarayil Subiar, Sushanth Ankush Mane, Irfan Ali Valiya peediyakkal, and under the guidance of Prof. Dr. Tim Weber, we developed a Python-based application that leverages the YOLOv8 model for highly accurate defect detection in casting parts.

Key insights:
 
  •	Benchmarking played a crucial role in validating the performance of our model. We systematically compared the YOLOv8 model against a well-documented CNN-based model, benchmarking key metrics such as precision, recall, F1-score, and overall accuracy.
  
  •	In addition to benchmarking, we tackled model drift by implementing data preprocessing techniques (adding Gaussian noise) to ensure model reliability over time. This ensures the model performs well even when image input quality fluctuates.
  
  •	YOLOv8, renowned for its speed and accuracy, was trained on a labeled dataset and achieved an impressive F1-score of 99.80%, classifying defective and non-defective castings with high precision.
  
  •	YOLOv8 exceeded expectations, achieving a 99.80% F1-score, 100% recall, 99.86% precision, and 99.61% accuracy, outperforming both the DenseNet model and YOLOv8 without drift. This confirms its robustness in detecting both defective and non-defective cast components, even in challenging real-world conditions where image quality varies.
  
  •	The comparison highlighted how YOLOv8's ability to handle model drift—by addressing deterioration in image quality—enhanced its stability, while the DenseNet model showed slightly lower performance.
  
  •	The application, built with Tkinter, provides a user-friendly interface, allowing users to easily upload images and get instant results on casting defects.

  Our project demonstrates that automated defect detection using YOLOv8 can significantly improve quality control in industries like automotive, aerospace, and construction, where precision is critical. The user-friendly interface, built with Tkinter, enables easy image 

📄Project Highlights:

  •	Python-based detection using YOLOv8
  
  •	Benchmarking to validate performance
  
  •	Advanced preprocessing to handle model drift
  
  •	User-friendly interface, built with Tkinter
  
📄 Key metrics:

  •	99.80% F1-score
  
  •	100% recall
  
  •	99.86% precision
  
  •	99.61% accuracy

This project was conducted at the Applied AI for Digital Production Management division of Deggendorf Institute of Technology Campus Cham, and we’re excited about its potential to transform industrial quality control processes.

Check out our work and code to explore how this solution can be applied to real-world quality control processes! 🌍
