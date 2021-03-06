/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.fyr_gui;

import com.mongodb.BasicDBObject;
import java.awt.Color;
import java.awt.Desktop;
import java.io.File;
import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Base64;
import java.util.Date;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.imageio.ImageIO;
import com.itextpdf.text.BaseColor;
import com.itextpdf.text.Chunk;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;
import com.mongodb.DBCursor;
import java.awt.HeadlessException;
import java.awt.Toolkit;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.Scanner;


public class ApplicationWindow extends javax.swing.JFrame {

    /**
     * Creates new form ApplicationWindow
     */
    private String dateTime;
    private String fileName;
    private int grade;
    private int modelType;
    private String imageBase64;
    private String imagePath;

    public ApplicationWindow() {
        super("Breast Cancer Grade Classification");
        initComponents();
        awPnlData.setVisible(false);
        awbtnProcess.setVisible(false);
        awlblms.setVisible(false);
        ddlModType.setVisible(false);
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
        awbtnHome = new javax.swing.JButton();
        awbtnApp = new javax.swing.JButton();
        awbtnSearch = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        awbtnUpload = new javax.swing.JButton();
        awbtnProcess = new javax.swing.JButton();
        jLabel2 = new javax.swing.JLabel();
        awlblImage = new javax.swing.JLabel();
        awPnlData = new javax.swing.JPanel();
        jLabel4 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        awlblfn = new javax.swing.JLabel();
        jLabel5 = new javax.swing.JLabel();
        awlbldt = new javax.swing.JLabel();
        jLabel7 = new javax.swing.JLabel();
        awlblgc = new javax.swing.JLabel();
        awbtnPrint = new javax.swing.JButton();
        apbtnSave = new javax.swing.JButton();
        awlblcid = new javax.swing.JLabel();
        ddlModType = new javax.swing.JComboBox<>();
        awlblms = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setBackground(new java.awt.Color(204, 255, 255));
        setIconImage(Toolkit.getDefaultToolkit().
            getImage("./images/logo.jpg"));
        setResizable(false);

        jPanel1.setBackground(new java.awt.Color(204, 255, 255));
        jPanel1.setForeground(new java.awt.Color(204, 255, 255));

        jPanel2.setBackground(new java.awt.Color(102, 204, 255));
        jPanel2.setForeground(new java.awt.Color(204, 255, 255));

        awbtnHome.setBackground(new java.awt.Color(102, 204, 255));
        awbtnHome.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        awbtnHome.setForeground(new java.awt.Color(102, 102, 102));
        awbtnHome.setText("Home");
        awbtnHome.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                awbtnHomeActionPerformed(evt);
            }
        });

        awbtnApp.setBackground(new java.awt.Color(102, 204, 255));
        awbtnApp.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        awbtnApp.setForeground(new java.awt.Color(102, 102, 102));
        awbtnApp.setText("Application");
        awbtnApp.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                awbtnAppActionPerformed(evt);
            }
        });

        awbtnSearch.setBackground(new java.awt.Color(102, 204, 255));
        awbtnSearch.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        awbtnSearch.setForeground(new java.awt.Color(102, 102, 102));
        awbtnSearch.setText("Search");
        awbtnSearch.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                awbtnSearchActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout jPanel2Layout = new javax.swing.GroupLayout(jPanel2);
        jPanel2.setLayout(jPanel2Layout);
        jPanel2Layout.setHorizontalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel2Layout.createSequentialGroup()
                .addGroup(jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                    .addComponent(awbtnSearch, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(awbtnApp, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, 360, Short.MAX_VALUE)
                    .addComponent(awbtnHome, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
        jPanel2Layout.setVerticalGroup(
            jPanel2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel2Layout.createSequentialGroup()
                .addGap(120, 120, 120)
                .addComponent(awbtnHome, javax.swing.GroupLayout.PREFERRED_SIZE, 87, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(awbtnApp, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(awbtnSearch, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(504, Short.MAX_VALUE))
        );

        jLabel1.setBackground(new java.awt.Color(204, 255, 255));
        jLabel1.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        jLabel1.setForeground(new java.awt.Color(102, 102, 102));
        jLabel1.setText("Grade Classification");

        awbtnUpload.setBackground(new java.awt.Color(102, 204, 255));
        awbtnUpload.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        awbtnUpload.setForeground(new java.awt.Color(102, 102, 102));
        awbtnUpload.setText("Upload");
        awbtnUpload.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                awbtnUploadActionPerformed(evt);
            }
        });

        awbtnProcess.setBackground(new java.awt.Color(102, 204, 255));
        awbtnProcess.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        awbtnProcess.setForeground(new java.awt.Color(102, 102, 102));
        awbtnProcess.setText("Process");
        awbtnProcess.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                awbtnProcessActionPerformed(evt);
            }
        });

        jLabel2.setBackground(new java.awt.Color(204, 255, 255));
        jLabel2.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jLabel2.setForeground(new java.awt.Color(102, 102, 102));
        jLabel2.setText("Browse Files:");

        awlblImage.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        awlblImage.setForeground(new java.awt.Color(102, 102, 102));
        awlblImage.setText("No file is choosen");
        awlblImage.setVerifyInputWhenFocusTarget(false);

        awPnlData.setBackground(new java.awt.Color(204, 255, 255));
        awPnlData.setForeground(new java.awt.Color(204, 255, 255));

        jLabel4.setBackground(new java.awt.Color(204, 255, 255));
        jLabel4.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jLabel4.setForeground(new java.awt.Color(102, 102, 102));
        jLabel4.setText("Record ID:");
        jLabel4.setToolTipText("");

        jLabel3.setBackground(new java.awt.Color(204, 255, 255));
        jLabel3.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jLabel3.setForeground(new java.awt.Color(102, 102, 102));
        jLabel3.setText("File name:");

        awlblfn.setBackground(new java.awt.Color(204, 255, 255));
        awlblfn.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        awlblfn.setForeground(new java.awt.Color(102, 102, 102));
        awlblfn.setText("test");

        jLabel5.setBackground(new java.awt.Color(204, 255, 255));
        jLabel5.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jLabel5.setForeground(new java.awt.Color(102, 102, 102));
        jLabel5.setText("Date time: ");

        awlbldt.setBackground(new java.awt.Color(204, 255, 255));
        awlbldt.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        awlbldt.setForeground(new java.awt.Color(102, 102, 102));
        awlbldt.setText("test");

        jLabel7.setBackground(new java.awt.Color(204, 255, 255));
        jLabel7.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jLabel7.setForeground(new java.awt.Color(102, 102, 102));
        jLabel7.setText("Grade Classification");

        awlblgc.setBackground(new java.awt.Color(204, 255, 255));
        awlblgc.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        awlblgc.setForeground(new java.awt.Color(102, 102, 102));
        awlblgc.setText("test");

        awbtnPrint.setBackground(new java.awt.Color(102, 204, 255));
        awbtnPrint.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        awbtnPrint.setForeground(new java.awt.Color(102, 102, 102));
        awbtnPrint.setText("Export to PDF");
        awbtnPrint.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                awbtnPrintActionPerformed(evt);
            }
        });

        apbtnSave.setBackground(new java.awt.Color(102, 204, 255));
        apbtnSave.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        apbtnSave.setForeground(new java.awt.Color(102, 102, 102));
        apbtnSave.setText("Save");
        apbtnSave.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                apbtnSaveActionPerformed(evt);
            }
        });

        awlblcid.setBackground(new java.awt.Color(204, 255, 255));
        awlblcid.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        awlblcid.setForeground(new java.awt.Color(102, 102, 102));
        awlblcid.setText("test");

        javax.swing.GroupLayout awPnlDataLayout = new javax.swing.GroupLayout(awPnlData);
        awPnlData.setLayout(awPnlDataLayout);
        awPnlDataLayout.setHorizontalGroup(
            awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(awPnlDataLayout.createSequentialGroup()
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(awbtnPrint, javax.swing.GroupLayout.DEFAULT_SIZE, 149, Short.MAX_VALUE)
                    .addComponent(jLabel7, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jLabel5, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jLabel3, javax.swing.GroupLayout.PREFERRED_SIZE, 143, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, awPnlDataLayout.createSequentialGroup()
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(apbtnSave, javax.swing.GroupLayout.PREFERRED_SIZE, 115, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(122, 122, 122))
                    .addGroup(awPnlDataLayout.createSequentialGroup()
                        .addGap(97, 97, 97)
                        .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(awlblfn, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(awlbldt, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(awlblgc, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(awlblcid, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addContainerGap())))
            .addGroup(awPnlDataLayout.createSequentialGroup()
                .addComponent(jLabel4, javax.swing.GroupLayout.PREFERRED_SIZE, 143, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );
        awPnlDataLayout.setVerticalGroup(
            awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(awPnlDataLayout.createSequentialGroup()
                .addContainerGap()
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel4, javax.swing.GroupLayout.DEFAULT_SIZE, 37, Short.MAX_VALUE)
                    .addComponent(awlblcid, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel3, javax.swing.GroupLayout.PREFERRED_SIZE, 36, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(awlblfn, javax.swing.GroupLayout.PREFERRED_SIZE, 36, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel5, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(awlbldt, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel7, javax.swing.GroupLayout.PREFERRED_SIZE, 36, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(awlblgc, javax.swing.GroupLayout.PREFERRED_SIZE, 36, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(32, 32, 32)
                .addGroup(awPnlDataLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(awbtnPrint, javax.swing.GroupLayout.PREFERRED_SIZE, 47, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(apbtnSave, javax.swing.GroupLayout.PREFERRED_SIZE, 47, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(22, 22, 22))
        );

        ddlModType.setBackground(new java.awt.Color(204, 255, 255));
        ddlModType.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        ddlModType.setForeground(new java.awt.Color(102, 102, 102));
        ddlModType.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "1", "2" }));

        awlblms.setBackground(new java.awt.Color(204, 255, 255));
        awlblms.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        awlblms.setForeground(new java.awt.Color(102, 102, 102));
        awlblms.setText("Model selection:");

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGap(58, 58, 58)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 208, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, jPanel1Layout.createSequentialGroup()
                                        .addComponent(awlblms, javax.swing.GroupLayout.PREFERRED_SIZE, 128, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                        .addComponent(ddlModType, javax.swing.GroupLayout.PREFERRED_SIZE, 112, javax.swing.GroupLayout.PREFERRED_SIZE))
                                    .addGroup(javax.swing.GroupLayout.Alignment.LEADING, jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                        .addComponent(awlblImage, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.PREFERRED_SIZE, 128, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addGroup(jPanel1Layout.createSequentialGroup()
                                            .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 247, javax.swing.GroupLayout.PREFERRED_SIZE)
                                            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                            .addComponent(awbtnUpload, javax.swing.GroupLayout.PREFERRED_SIZE, 112, javax.swing.GroupLayout.PREFERRED_SIZE))))
                                .addGap(79, 79, 79)
                                .addComponent(awbtnProcess, javax.swing.GroupLayout.PREFERRED_SIZE, 112, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addContainerGap(134, Short.MAX_VALUE))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGap(12, 12, 12)
                        .addComponent(awPnlData, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addContainerGap())))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGap(111, 111, 111)
                .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(27, 27, 27)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 41, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(awbtnUpload))
                .addGap(18, 18, 18)
                .addComponent(awlblImage, javax.swing.GroupLayout.PREFERRED_SIZE, 128, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(awlblms, javax.swing.GroupLayout.PREFERRED_SIZE, 39, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(ddlModType, javax.swing.GroupLayout.PREFERRED_SIZE, 39, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(awbtnProcess))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(awPnlData, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, 713, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void awbtnHomeActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_awbtnHomeActionPerformed
        new HomeWindow().setVisible(true);
        this.setVisible(false);
    }//GEN-LAST:event_awbtnHomeActionPerformed

    private void awbtnUploadActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_awbtnUploadActionPerformed
        try {
            JFileChooser file = new JFileChooser();
            file.setCurrentDirectory(new File(System.getProperty("user.home")));
            FileNameExtensionFilter filter = new FileNameExtensionFilter("*.Images", "jpg", "png");
            file.addChoosableFileFilter(filter);
            int result = file.showSaveDialog(null);
            if (result == JFileChooser.APPROVE_OPTION) {
                File selectedFile = file.getSelectedFile();
                ImageIcon img1 = new ImageIcon(selectedFile.getAbsolutePath());
                Image img = img1.getImage();
                String regex = "(.*/)*.+\\.(png|jpg|jpeg|PNG|JPG|JPEG)$";
                Pattern p = Pattern.compile(regex);
                Matcher m = p.matcher(selectedFile.getName());
                if (m.matches() && img.getHeight(file) >= 128 && img.getWidth(file) >= 128) {
                    awlblImage.setIcon(ResizeImage(selectedFile.getAbsolutePath()));
                    imagePath = selectedFile.getAbsolutePath();
                    fileName = selectedFile.getName();
                    BufferedImage bImage = ImageIO.read(selectedFile);
                    Image scaledImage = bImage.getScaledInstance(awlblImage.getWidth(), awlblImage.getHeight(), Image.SCALE_SMOOTH);
                    BufferedImage imageBuff = new BufferedImage(awlblImage.getWidth(), awlblImage.getHeight(), BufferedImage.TYPE_INT_RGB);
                    imageBuff.getGraphics().drawImage(scaledImage, 0, 0, new Color(0, 0, 0), null);
                    awlblms.setVisible(true);
                    ddlModType.setVisible(true);
                    awbtnProcess.setVisible(true);
                    ByteArrayOutputStream bos = new ByteArrayOutputStream();
                    String extension = "";
                    int i = selectedFile.getName().lastIndexOf('.');
                    if (i > 0) {
                        extension = selectedFile.getName().substring(i + 1);
                    }
                    ImageIO.write(imageBuff, extension, bos);
                    byte[] data = bos.toByteArray();
                    imageBase64 = Base64.getEncoder().encodeToString(data);
                } else {
                    awlblImage.setText("No file is choosen");
                    awlblImage.setIcon(null);
                    awPnlData.setVisible(false);
                    awlblms.setVisible(false);
                    ddlModType.setVisible(false);
                    awbtnProcess.setVisible(false);
                    if (!m.matches()) {
                        JOptionPane.showMessageDialog(null, "Invalid file format! Allows only .png, .jpg and .jpeg", "Error", JOptionPane.ERROR_MESSAGE);
                    } else if (img.getHeight(file) < 128) {
                        JOptionPane.showMessageDialog(null, "Invalid image dimensions! Minimun hight is 128", "Error", JOptionPane.ERROR_MESSAGE);
                    } else if (img.getWidth(file) < 128) {
                        JOptionPane.showMessageDialog(null, "Invalid image dimensions! Minimun width is 128", "Error", JOptionPane.ERROR_MESSAGE);
                    } else if (img.getWidth(file) < 128 && img.getHeight(file) < 128) {
                        JOptionPane.showMessageDialog(null, "Invalid image dimensions! Minimun width and height are 128", "Error", JOptionPane.ERROR_MESSAGE);
                    } else {
                        JOptionPane.showMessageDialog(null, "Invalid Input!", "Error", JOptionPane.ERROR_MESSAGE);
                    }
                }
            }
        } catch (HeadlessException | IOException ex) {
            System.out.println(ex);
            JOptionPane.showMessageDialog(null, "An error occured while processing the input image!", "Error", JOptionPane.ERROR_MESSAGE);
            java.util.logging.Logger.getLogger(ApplicationWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_awbtnUploadActionPerformed

    private void awbtnAppActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_awbtnAppActionPerformed

    }//GEN-LAST:event_awbtnAppActionPerformed

    private void awbtnSearchActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_awbtnSearchActionPerformed
        new SearchWindow().setVisible(true);
        this.setVisible(false);
    }//GEN-LAST:event_awbtnSearchActionPerformed

    private void awbtnProcessActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_awbtnProcessActionPerformed
        try {
            awPnlData.setVisible(false);
            APIConnction apiConn = new APIConnction();
            modelType = Integer.parseInt((String) ddlModType.getSelectedItem());
            int predict = apiConn.getGrade(imagePath, modelType);
            if (predict != -1) {                
                String data = null;
                String path="count_id.txt";
                FileReader fr = new FileReader(path);
                Scanner scan = new Scanner(fr);
                while (scan.hasNextLine()) {
                      data = scan.nextLine();
                }  
                int count_id=Integer.parseInt(data)+1;
                try (PrintWriter ptr = new PrintWriter (path)) {
                    ptr.write(String.valueOf(count_id));
                    ptr.close();
                }
                grade = predict;
                awPnlData.setVisible(true);
                awlblfn.setText(fileName);
                dateTime = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss").format(new Date());
                awlblcid.setText(String.valueOf(count_id));
                awlbldt.setText(dateTime);
                awlblgc.setText(String.valueOf(predict));
            } else {
                awPnlData.setVisible(false);
                JOptionPane.showMessageDialog(null, "An error occured while processing!", "Error", JOptionPane.ERROR_MESSAGE);
            }
        } catch (HeadlessException | NumberFormatException | FileNotFoundException ex) {
            System.out.println(ex);
            awPnlData.setVisible(false);
            JOptionPane.showMessageDialog(null, "An error occured while processing!", "Error", JOptionPane.ERROR_MESSAGE);
            java.util.logging.Logger.getLogger(ApplicationWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_awbtnProcessActionPerformed

    private void apbtnSaveActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_apbtnSaveActionPerformed
        try {
            String data = null;
            String path="count_id.txt";
            FileReader fr = new FileReader(path);
            Scanner scan = new Scanner(fr);
            while (scan.hasNextLine()) 
                  data = scan.nextLine();          
            BasicDBObject ob = new BasicDBObject();
            ob.put("date_time", dateTime);
            DBCursor cursor = Connector.con.find(ob);
            String st = null;
            while (cursor.hasNext()) {
                st = cursor.next().toString();
            }
            if (st == null) {          
                BasicDBObject doc = new BasicDBObject();
                doc.put("count_id", Integer.parseInt(data));
                doc.put("date_time", dateTime);
                doc.put("grade", grade);
                doc.put("file_name", fileName);
                doc.put("model_type", modelType);
                doc.put("image_base64", imageBase64);
                Connector.con.insert(doc);
                JOptionPane.showMessageDialog(null, "Successfully saved the record in to the database! The transfer ID of the saved record is "
                        + data);                
            } else {
                JOptionPane.showMessageDialog(null, "The record already exsist", "Error",
                            JOptionPane.ERROR_MESSAGE);
            }
        } catch (HeadlessException | FileNotFoundException | NumberFormatException ex) {
            System.out.println(ex);
            JOptionPane.showMessageDialog(null, "An error occured while Saving!", "Error",
                    JOptionPane.ERROR_MESSAGE);
            java.util.logging.Logger.getLogger(ApplicationWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_apbtnSaveActionPerformed

    private void awbtnPrintActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_awbtnPrintActionPerformed
        try {
            String count_id = null;
            String path="count_id.txt";
            FileReader fr = new FileReader(path);
            Scanner scan = new Scanner(fr);
            while (scan.hasNextLine()) 
                  count_id = scan.nextLine();          
            String st = "./reports/"+"report_" + dateTime.replace("-", "").replace(" ", "").replace(":", "") + ".pdf";
            Document doc = new Document();
            PdfWriter.getInstance(doc, new FileOutputStream(st));
            doc.open();
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("Breast Cancer Grade Classification Report",
                    new Font(Font.FontFamily.HELVETICA, 18.0f, Font.BOLD, BaseColor.BLACK)));
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("Image :", new Font(Font.FontFamily.HELVETICA, 12.0f, Font.NORMAL, BaseColor.BLACK)));
            byte[] imageBytes = javax.xml.bind.DatatypeConverter.parseBase64Binary(imageBase64);
            com.itextpdf.text.Image img = com.itextpdf.text.Image.getInstance(imageBytes);
            doc.add(img);
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("Record ID :", new Font(Font.FontFamily.HELVETICA, 12.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(new Paragraph(count_id, new Font(Font.FontFamily.HELVETICA, 10.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("File Name :", new Font(Font.FontFamily.HELVETICA, 12.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(new Paragraph(fileName, new Font(Font.FontFamily.HELVETICA, 10.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("Selected Model :", new Font(Font.FontFamily.HELVETICA, 12.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(new Paragraph(String.valueOf(modelType), new Font(Font.FontFamily.HELVETICA, 10.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("Executed date and time :", new Font(Font.FontFamily.HELVETICA, 12.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(new Paragraph(dateTime, new Font(Font.FontFamily.HELVETICA, 10.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(Chunk.NEWLINE);
            doc.add(new Paragraph("Grade :", new Font(Font.FontFamily.HELVETICA, 12.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.add(new Paragraph(String.valueOf(grade), new Font(Font.FontFamily.HELVETICA, 10.0f, Font.NORMAL, BaseColor.BLACK)));
            doc.close();
            File f = new File(st);
            Desktop.getDesktop().open(f);
        } catch (DocumentException | IOException ex) {
            System.out.println(ex);
            if ("java.io.FileNotFoundException".equals(ex.getClass().getName())) {
                JOptionPane.showMessageDialog(null, "The file is already opened!", "Warninig",
                        JOptionPane.WARNING_MESSAGE);
            } else {
                JOptionPane.showMessageDialog(null, "An error Occured while exporting in to PDF format!", "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
            java.util.logging.Logger.getLogger(ApplicationWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }//GEN-LAST:event_awbtnPrintActionPerformed

    public ImageIcon ResizeImage(String ImagePath) {
        ImageIcon img1 = new ImageIcon(ImagePath);
        Image img = img1.getImage();
        Image newImg = img.getScaledInstance(awlblImage.getWidth(), awlblImage.getHeight(), Image.SCALE_SMOOTH);
        ImageIcon image = new ImageIcon(newImg);
        return image;
    }

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
            java.util.logging.Logger.getLogger(ApplicationWindow.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(() -> {
            new ApplicationWindow().setVisible(true);
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton apbtnSave;
    private javax.swing.JPanel awPnlData;
    private javax.swing.JButton awbtnApp;
    private javax.swing.JButton awbtnHome;
    private javax.swing.JButton awbtnPrint;
    private javax.swing.JButton awbtnProcess;
    private javax.swing.JButton awbtnSearch;
    private javax.swing.JButton awbtnUpload;
    private javax.swing.JLabel awlblImage;
    private javax.swing.JLabel awlblcid;
    private javax.swing.JLabel awlbldt;
    private javax.swing.JLabel awlblfn;
    private javax.swing.JLabel awlblgc;
    private javax.swing.JLabel awlblms;
    private javax.swing.JComboBox<String> ddlModType;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JLabel jLabel5;
    private javax.swing.JLabel jLabel7;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanel2;
    // End of variables declaration//GEN-END:variables
}
