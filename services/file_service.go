package services

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

func SaveUploadedFile(c *gin.Context) (string, error) {
	file, err := c.FormFile("file")
	bucketId := c.PostForm("bucket_id")
	fileKey := c.PostForm("file_key")

	if err != nil {
		return "", fmt.Errorf("failed to retrieve file: %v", err)
	}

	if bucketId == "" {
		return "", fmt.Errorf("missing required field: bucket_id")
	}

	if fileKey == "" {
		fileKey = uuid.New().String()
	}

	fileExt := filepath.Ext(file.Filename)
	if fileExt == "" {
		return "", fmt.Errorf("uploaded file has no extension")
	}

	storedFileName := fileKey + fileExt

	uploadDir := filepath.Join("uploads", bucketId)
	if err := c.SaveUploadedFile(file, filepath.Join(uploadDir, storedFileName)); err != nil {
		return "", fmt.Errorf("failed to save uploaded file: %v", err)
	}

	return fileKey, nil
}

func FetchFilePath(fileKey string) (string, error) {
	filePath := filepath.Join("uploads", fileKey)

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return "", fmt.Errorf("file not found")
	}

	return filePath, nil
}
