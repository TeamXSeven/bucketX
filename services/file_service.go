package services

import (
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/simple"
	"gorgonia.org/tensor"
)

func SaveUploadedFile(c *gin.Context) (tensor.Tensor, string, error) {
	file, err := c.FormFile("file")
	bucketId := c.PostForm("bucket_id")
	if err != nil {
		return nil, "", fmt.Errorf("failed to retrieve file: %v", err)
	}

	if bucketId == "" {
		return nil, "", fmt.Errorf("bucket_id is required")
	}

	filename := filepath.Base(file.Filename)
	uuidFilename := uuid.New().String() + "-" + filename

	filePath := filepath.Join("uploads", bucketId, uuidFilename)

	if err := c.SaveUploadedFile(file, filePath); err != nil {
		return nil, "", fmt.Errorf("failed to save file: %v", err)
	}

	tensor, err := preprocessImage(filePath)

	if err != nil {
		return nil, "", fmt.Errorf("failed to preprocess image: %v", err)
	}

	return tensor, uuidFilename, nil
}

func preprocessImage(filePath string) (tensor.Tensor, error) {
	uploadedFile, err := os.Open(filePath)
	if err != nil {
		fmt.Printf("failed to open uploaded file: %v", err)
		return nil, err
	}
	defer uploadedFile.Close()

	img, _, er := image.Decode(uploadedFile)

	if er != nil {
		fmt.Printf("failed to decode image: %v", err)
		return nil, err
	}

	rgbaImg := image.NewRGBA(img.Bounds())

	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			rgbaImg.Set(x, y, img.At(x, y))
		}
	}

	imageData := make([]float32, 224*224*3)

	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			r, g, b, _ := rgbaImg.At(x, y).RGBA()
			imageData[(y*224+x)*3] = float32(r) / 65535.0
			imageData[(y*224+x)*3+1] = float32(g) / 65535.0
			imageData[(y*224+x)*3+2] = float32(b) / 65535.0
		}
	}

	t := tensor.New(tensor.WithShape(1, 3, 224, 224), tensor.WithBacking(imageData))

	return saveToVectorDB(t)
}

func saveToVectorDB(tensor tensor.Tensor) (tensor.Tensor, error) {
	backend := simple.NewSimpleGraph()
	model := onnx.NewModel(backend)
	b, _ := os.ReadFile("ResNet50.onnx")

	err := model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}

	model.SetInput(0, tensor)

	output, _ := model.GetOutputTensors()

	return output[0], nil
}

func FetchFilePath(filename string, bucketId string) (string, error) {
	filePath := filepath.Join("uploads", bucketId, filename)

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "", fmt.Errorf("file does not exist: %v", err)
	} else if err != nil {
		return "", fmt.Errorf("failed to retrieve file info: %v", err)
	}

	return filePath, nil
}
