/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.fyr_gui;


public class APIInput {

    public int modelType;
    public String imagePath;

    public APIInput(int modelType, String imagePath) {
        this.modelType = modelType;
        this.imagePath = imagePath;
    }

    public void setModelType(int modelType) {
        this.modelType = modelType;
    }

    public void setImageBase64(String imageBase64) {
        this.imagePath = imageBase64;
    }

    public int getModelType() {
        return modelType;
    }

    public String getImageBase64() {
        return imagePath;
    }

}
