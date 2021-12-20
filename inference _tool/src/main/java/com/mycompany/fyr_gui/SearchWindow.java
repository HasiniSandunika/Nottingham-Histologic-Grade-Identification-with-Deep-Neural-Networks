/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.fyr_gui;

import com.mongodb.BasicDBObject;
import com.mongodb.DBCursor;
import java.awt.HeadlessException;
import javax.swing.JOptionPane;
import org.json.JSONObject;
import java.awt.Toolkit;
import org.json.JSONException;


public class SearchWindow extends javax.swing.JFrame {

    /**
     * Creates new form SearchWindow
     */
    public SearchWindow() {
        super( "Breast Cancer Grade Classification" );
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        jPanel2 = new javax.swing.JPanel();
        swbtnHome = new javax.swing.JButton();
        swbtnApp = new javax.swing.JButton();
        swbtnSearch = new javax.swing.JButton();
        swbtnSearchDB = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        transferId = new javax.swing.JTextField();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setBackground(new java.awt.Color(204, 255, 255));
        setIconImage(Toolkit.getDefaultToolkit().
            getImage("./images/logo.jpg"));
        setResizable(false);

        jPanel1.setBackground(new java.awt.Color(204, 255, 255));
        jPanel1.setForeground(new java.awt.Color(204, 255, 255));

        jPanel2.setBackground(new java.awt.Color(102, 204, 255));
        jPanel2.setForeground(new java.awt.Color(102, 204, 255));

        swbtnHome.setBackground(new java.awt.Color(102, 204, 255));
        swbtnHome.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        swbtnHome.setForeground(new java.awt.Color(102, 102, 102));
        swbtnHome.setText("Home");
        swbtnHome.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                swbtnHomeActionPerformed(evt);
            }
        });

        swbtnApp.setBackground(new java.awt.Color(102, 204, 255));
        swbtnApp.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        swbtnApp.setForeground(new java.awt.Color(102, 102, 102));
        swbtnApp.setText("Application");
        swbtnApp.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                swbtnAppActionPerformed(evt);
            }
        });

        swbtnSearch.setBackground(new java.awt.Color(102, 204, 255));
        swbtnSearch.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        swbtnSearch.setForeground(new java.awt.Color(102, 102, 102));
        swbtnSearch.setText("Search");
        swbtnSearch.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                swbtnSearchActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(swbtnSearch, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(swbtnApp, javax.swing.GroupLayout.DEFAULT_SIZE, 371, Short.MAX_VALUE)
                    .addComponent(swbtnHome, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(93, 93, 93)
                .addComponent(swbtnHome, javax.swing.GroupLayout.PREFERRED_SIZE, 83, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(swbtnApp, javax.swing.GroupLayout.PREFERRED_SIZE, 85, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(swbtnSearch, javax.swing.GroupLayout.PREFERRED_SIZE, 84, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(430, Short.MAX_VALUE))
        );

        swbtnSearchDB.setBackground(new java.awt.Color(102, 204, 255));
        swbtnSearchDB.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        swbtnSearchDB.setForeground(new java.awt.Color(102, 102, 102));
        swbtnSearchDB.setText("Search");
        swbtnSearchDB.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                swbtnSearchDBActionPerformed(evt);
            }
        });

        jLabel1.setBackground(new java.awt.Color(204, 255, 255));
        jLabel1.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jLabel1.setForeground(new java.awt.Color(102, 102, 102));
        jLabel1.setText("Search by Transfer ID :");

        transferId.setBackground(new java.awt.Color(204, 255, 255));
        transferId.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        transferId.setForeground(new java.awt.Color(102, 102, 102));

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 154, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(transferId, javax.swing.GroupLayout.PREFERRED_SIZE, 294, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 64, Short.MAX_VALUE)
                .addComponent(swbtnSearchDB, javax.swing.GroupLayout.PREFERRED_SIZE, 142, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(115, 115, 115))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGap(145, 145, 145)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(swbtnSearchDB, javax.swing.GroupLayout.PREFERRED_SIZE, 55, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(transferId, javax.swing.GroupLayout.PREFERRED_SIZE, 44, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void swbtnHomeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_swbtnHomeActionPerformed
        new HomeWindow().setVisible(true);
        this.setVisible(false);
    }//GEN-LAST:event_swbtnHomeActionPerformed

    private void swbtnAppActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_swbtnAppActionPerformed
        new ApplicationWindow().setVisible(true);
        this.setVisible(false);
    }//GEN-LAST:event_swbtnAppActionPerformed

    private void swbtnSearchActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_swbtnSearchActionPerformed

    }//GEN-LAST:event_swbtnSearchActionPerformed

    private void swbtnSearchDBActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_swbtnSearchDBActionPerformed
        try {
            String str;
                str=transferId.getText();
            BasicDBObject ob = new BasicDBObject();
            ob.put("count_id", Integer.parseInt(str));
            DBCursor cursor = Connector.con.find(ob);
            String st = null;
            while (cursor.hasNext()) {
                st = cursor.next().toString();
            }
            if (st != null) {
                JSONObject jsonObj = new JSONObject(st);
                new SearchDataWindow(jsonObj.getInt("count_id"), jsonObj.getString("date_time"), jsonObj.getString("file_name"), jsonObj.getInt("grade"), jsonObj.getInt("model_type"),
                        jsonObj.getString("image_base64")).setVisible(true);
                this.setVisible(false);
            } else {
                JOptionPane.showMessageDialog(null, "No such record is found with transfer ID " + transferId.getText(), "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        } catch (HeadlessException | NumberFormatException | JSONException ex) {
            System.out.println(ex);
            if ("java.lang.NumberFormatException".equals(ex.getClass().getName())){
                JOptionPane.showMessageDialog(null, "Invalid transfer ID", "Error",
                        JOptionPane.ERROR_MESSAGE);
                java.util.logging.Logger.getLogger(SearchWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
            }                
            else{
                JOptionPane.showMessageDialog(null, "An error occured while retreiving the record!", "Error",
                    JOptionPane.ERROR_MESSAGE);
            java.util.logging.Logger.getLogger(SearchWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
            }                       
        }
    }//GEN-LAST:event_swbtnSearchDBActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(SearchWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(() -> {
            new SearchWindow().setVisible(true);
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel jLabel1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JButton swbtnApp;
    private javax.swing.JButton swbtnHome;
    private javax.swing.JButton swbtnSearch;
    private javax.swing.JButton swbtnSearchDB;
    private javax.swing.JTextField transferId;
    // End of variables declaration//GEN-END:variables
}
