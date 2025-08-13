# Email Notification Setup Guide

This guide will help you set up email notifications for your FinRisk project.

## 🎯 **What You'll Get**

✅ **Deployment Notifications** - When your app deploys to staging/production
✅ **ML Training Alerts** - When models are trained and their performance
✅ **System Alerts** - When something goes wrong or needs attention

## 📧 **Step 1: Set Up Gmail App Password**

Since you're using `rfondufe@gmail.com`, you need to create an "App Password":

1. **Go to your Google Account settings:**
   - Visit: https://myaccount.google.com/
   - Sign in with `rfondufe@gmail.com`

2. **Enable 2-Factor Authentication (if not already enabled):**
   - Go to "Security" → "2-Step Verification"
   - Follow the setup process

3. **Create App Password:**
   - Go to "Security" → "App passwords"
   - Select "Mail" and "Other (Custom name)"
   - Name it "FinRisk Notifications"
   - Copy the 16-character password (like: `abcd efgh ijkl mnop`)

## 🔧 **Step 2: Add Email Secrets to GitHub**

1. **Go to your GitHub repository:**
   - Visit: https://github.com/kovikov/AMDARI_PROJECT_FINANCIAL_RISK
   - Click "Settings" → "Secrets and variables" → "Actions"

2. **Add these secrets:**
   ```
   EMAIL_SMTP_SERVER = smtp.gmail.com
   EMAIL_SMTP_PORT = 587
   EMAIL_USERNAME = rfondufe@gmail.com
   EMAIL_PASSWORD = [Your 16-character app password from Step 1]
   NOTIFICATION_EMAIL = rfondufe@gmail.com
   ```

## 🚀 **Step 3: Test the Setup**

1. **Make a small change to trigger a deployment:**
   ```bash
   # Add a comment to any file
   echo "# Test notification" >> finrisk/README.md
   git add .
   git commit -m "Test email notifications"
   git push origin main
   ```

2. **Check your email** - You should receive a deployment notification!

## 📋 **What You'll Receive**

### **Deployment Emails**
- **Subject:** `[FinRisk] Deployment Success - Production`
- **Content:** Environment, status, timestamp, branch, commit details
- **When:** Every time you deploy to staging or production

### **ML Training Emails**
- **Subject:** `[FinRisk] ML Training Completed - Credit Risk`
- **Content:** Model performance metrics, training duration, data size
- **When:** Every Sunday at 2 AM UTC (or manual trigger)

### **System Alert Emails**
- **Subject:** `[FinRisk] System Alert [HIGH] - Performance`
- **Content:** Alert type, severity, component, error details
- **When:** When system issues are detected

## 🔔 **GitHub Repository Notifications (Bonus)**

For additional notifications, set up GitHub repository notifications:

1. **Go to your repository settings:**
   - Visit: https://github.com/kovikov/AMDARI_PROJECT_FINANCIAL_RISK/settings/notifications

2. **Add your email:**
   - Click "Add email address"
   - Add: `rfondufe@gmail.com`
   - Verify the email

3. **Choose notification preferences:**
   - ✅ Workflow runs (success/failure)
   - ✅ Pull requests
   - ✅ Security alerts
   - ✅ Repository updates

## 🛠️ **Troubleshooting**

### **Email Not Sending?**
1. **Check app password:** Make sure you copied the 16-character password correctly
2. **Check GitHub secrets:** Verify all secrets are added correctly
3. **Check workflow logs:** Go to Actions tab to see error messages

### **Too Many Emails?**
You can adjust notification frequency by:
1. **Modifying workflow triggers** in `.github/workflows/`
2. **Setting up email filters** in Gmail
3. **Using different email addresses** for different notification types

### **Want Different Email?**
Simply change the `NOTIFICATION_EMAIL` secret to any other email address.

## 📱 **Mobile Notifications**

For mobile notifications, you can:
1. **Enable Gmail push notifications** on your phone
2. **Set up email filters** to prioritize FinRisk emails
3. **Use Gmail's "Important" feature** to highlight these emails

## 🎉 **You're All Set!**

Once you complete these steps, you'll receive:
- ✅ **Instant deployment notifications**
- ✅ **Weekly ML training reports**
- ✅ **System alerts when needed**
- ✅ **Repository activity updates**

Your FinRisk project will now keep you informed about everything important happening with your financial risk assessment system!

---

*Need help? Check the troubleshooting section or create an issue in the repository.*
