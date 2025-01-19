package services

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"crypto/sha256"

	"github.com/gin-gonic/gin"
)

type FileDataObject struct {
	BucketId string
	FileKey  string
	Filename string
	Hash     string
	File     []byte
}

var fileMetadataMap = make(map[string]FileDataObject)

func SaveUploadedFile(c *gin.Context) (string, string, error) {
	file, err := c.FormFile("file")
	bucketId := c.PostForm("bucket_id")
	fileKey := c.PostForm("file_key")
	if err != nil {
		return "", "", fmt.Errorf("failed to get uploaded file: %v", err)
	}

	if bucketId == "" || fileKey == "" {
		missingFields := []string{}
		if bucketId == "" {
			missingFields = append(missingFields, "bucket_id")
		}
		if fileKey == "" {
			missingFields = append(missingFields, "file_key")
		}
		return "", "", fmt.Errorf("missing required fields: %v", missingFields)
	}

	fileContent, err := file.Open()
	if err != nil {
		return "", "", fmt.Errorf("failed to open file: %v", err)
	}
	defer fileContent.Close()

	fileBytes, readErr := io.ReadAll(fileContent)
	if readErr != nil {
		return "", "", fmt.Errorf("failed to read file content: %v", readErr)
	}

	h := sha256.New()
	if _, hashErr := h.Write(fileBytes); hashErr != nil {
		return "", "", fmt.Errorf("failed to compute hash: %v", hashErr)
	}

	hashHex := fmt.Sprintf("%x", h.Sum(nil))

	if isDuplicate, checkErr := checkDuplicateHash(bucketId, hashHex); checkErr != nil {
		return "", "", fmt.Errorf("failed to check for duplicate hash: %v", checkErr)
	} else if isDuplicate {
		return "", "", fmt.Errorf("file with the same content already exists")
	}

	filename := filepath.Base(file.Filename)

	fileObject := FileDataObject{
		BucketId: bucketId,
		FileKey:  fileKey,
		Filename: filename,
		Hash:     hashHex,
		File:     fileBytes,
	}

	fileMetadataMap[fileKey] = fileObject

	if err := saveMetadataMapToFile(); err != nil {
		return "", "", fmt.Errorf("failed to save metadata map: %v", err)
	}

	return fileKey, filename, nil
}

func checkDuplicateHash(bucketId string, hash string) (bool, error) {
	for _, fileObject := range fileMetadataMap {
		if fileObject.BucketId == bucketId && fileObject.Hash == hash {
			return true, nil
		}
	}
	return false, nil
}

func FetchFileContent(fileKey string) ([]byte, string, error) {
	fileObject, exists := fileMetadataMap[fileKey]
	if !exists {
		return nil, "", fmt.Errorf("file with key %s does not exist", fileKey)
	}

	return fileObject.File, fileObject.Filename, nil
}

func LoadMetadataMapFromFile() error {
	file, err := os.Open("file_metadata.json")
	if err != nil {
		if os.IsNotExist(err) {
			fileMetadataMap = make(map[string]FileDataObject)
			return nil
		}
		return fmt.Errorf("failed to open metadata file: %v", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&fileMetadataMap); err != nil {
		return fmt.Errorf("failed to decode metadata map: %v", err)
	}

	return nil
}

func saveMetadataMapToFile() error {
	file, err := os.Create("file_metadata.json")
	if err != nil {
		return fmt.Errorf("failed to create metadata file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(fileMetadataMap); err != nil {
		return fmt.Errorf("failed to encode metadata map: %v", err)
	}

	return nil
}
