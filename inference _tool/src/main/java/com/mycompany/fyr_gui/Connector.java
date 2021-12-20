/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.fyr_gui;

import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.MongoClient;


public class Connector {
    public static DBCollection con;
     public static void connect(){
         try{
              MongoClient client=new MongoClient("localhost",27017);
              DB db = client.getDB("fyr_db");
              con= db.getCollection("fyr_collection");
         }
        catch(Exception ex){
            con=null;
            System.out.println(ex);           
            java.util.logging.Logger.getLogger(Connector.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }         
     }
}
