import cv2
fs = cv2.FileStorage("label.xml", cv2.FILE_STORAGE_READ)
fn_label_matrix = fs.getNode("label_matrix")
fn_label_number = fs.getNode("label_number")
label_matrix = fn_label_matrix.mat()
label_number = fn_label_number.real()
print(label_matrix)
print(label_number)