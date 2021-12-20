/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.fyr_gui;

import com.google.gson.Gson;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.json.JSONObject;


public class APIConnction {

    private static final String URL = "http://localhost:5000/prediction";

    public int getGrade(String imagePath, int modelType) {
        int predict = -1;
        try {
            HttpClient httpclient = HttpClients.createDefault();
            HttpPost httppost = new HttpPost(URL);
            APIInput apiInput = new APIInput(modelType, imagePath);
            Gson g = new Gson();
            String requestBody = g.toJson(apiInput);
            StringEntity params = new StringEntity(requestBody);
            httppost.addHeader("content-type", "application/json");
            httppost.setEntity(params);
            System.out.println(requestBody);
            HttpResponse response = httpclient.execute(httppost);
            HttpEntity entity = response.getEntity();
            String content = EntityUtils.toString(entity);
            System.out.println(content);
            JSONObject jsonObj = new JSONObject(content);
            if (entity != null && jsonObj.getInt("predict") != -1 && jsonObj.getBoolean("success")==true && response.getStatusLine().getStatusCode()==200) {
                predict = jsonObj.getInt("predict");              
            } else {
                predict = -1;
            }
        } catch (Exception ex) {
            System.out.println(ex);
            Logger.getLogger(APIConnction.class.getName()).log(Level.SEVERE, null, ex);
        }
        return predict;
    }
    
}
