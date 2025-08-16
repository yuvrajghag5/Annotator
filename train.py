# from ultralytics import YOLO
# import torch

# def main(): 
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     data_yaml='data.yaml'
#     model_yaml='model.yaml'
#     epochs= 100
#     imgsz= 416
#     batch= 8
#     # device='cpu',  # Use 'cpu' or '0' / '1' for GPU
#     project_name='multi-object-detect'
#     # workers= 2
#     # patience= 20
#     # cos_lr=True,
#     # optimizer='SGD',
#     # amp=True

#     # Initialize model from architecture YAML (no pre-trained weights)
#     model = YOLO(model_yaml)
    
#     # Train the model
#     model.train(
#         data=data_yaml,
#         epochs=epochs,
#         imgsz=imgsz,
#         batch=batch,
#         device='0',
#         project=project_name,
#         name='from_scratch',
#         workers = 4,
#         patience= 50,
#         cos_lr=True,
#         optimizer='Adamw',
#         lr0 = 0.01,
#         momentum = 0.937,
#         weight_decay = 0.0005,
#         amp=True,
#         save = True,
#         save_period = 5,
#         # exit_ok = True,
#         verbose = True,
#         close_mosaic=0,  
#         resume = True
#     )

#     print(f"\n✅ Training complete. Best weights saved in your folder")

# if __name__ == "__main__":
#     main()



from ultralytics import YOLO
import torch

def main(): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_yaml='data.yaml'
    model_yaml=r'C:\Users\yuvra\Documents\auto annotate\multi-object-detect\from_scratch\weights\last.pt'
    epochs= 500
    imgsz= 320
    batch= 8
    # device='cpu',  # Use 'cpu' or '0' / '1' for GPU
    project_name='multi-object-detect'
    # workers= 2
    # patience= 20
    # cos_lr=True,
    # optimizer='SGD',
    # amp=True

    # Initialize model from architecture YAML (no pre-trained weights)
    model = YOLO(model_yaml)
    
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_name,
        name='from_scratch',
        workers = 8,
        patience= 50,
        cos_lr=True,
        optimizer='SGD',
        lr0 = 0.01,
        momentum = 0.937,
        weight_decay = 0.0001,
        amp=True,
        save = True,
        save_period = 2,
        # exit_ok = True,
        verbose = False,
        close_mosaic=0,  
        resume = True
    )

    print(f"\n✅ Training complete. Best weights saved in your folder")
     # Run validation to get metrics
    print("\n🔍 Running validation to get Accuracy, Precision, Recall...")
    metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device)
    
    # Display metrics
    precision = metrics.results_dict.get('metrics/precision(B)', None)
    recall = metrics.results_dict.get('metrics/recall(B)', None)
    mAP50 = metrics.results_dict.get('metrics/mAP50(B)', None)
    mAP50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', None)

    print(f"\n📊 Evaluation Metrics:")
    print(f" Precision: {precision:.4f}" if precision is not None else " Precision: N/A")
    print(f" Recall:    {recall:.4f}" if recall is not None else " Recall: N/A")
    print(f" mAP@0.5:   {mAP50:.4f}" if mAP50 is not None else " mAP@0.5: N/A")
    print(f" mAP@0.5:0.95: {mAP50_95:.4f}" if mAP50_95 is not None else " mAP@0.5:0.95: N/A")


if __name__ == "__main__":
    main()