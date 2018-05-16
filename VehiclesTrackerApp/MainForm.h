#pragma once
#include <msclr\marshal_cppstd.h>
#include <Windows.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include <chrono>
#include "ctime"
#include "opencv2/opencv.hpp"

#include "BlockArray.h"
#include "StMrf.h"
#include "Tracking.h"
#include "Utils.h"
#include "NightDetection.h"
#include "Tracker.h"

using namespace Tracking;

namespace VehiclesTrackerApp {
	static const size_t NA_VALUE = std::numeric_limits<size_t>::max();
	struct Params
	{
		int background_init_length = 300;
		int frame_delay = 30;
		double background_update_weight = 0.05;
		size_t block_width = 16;
		size_t block_height = 20;
		double foreground_threshold = 0.05;
		int frame_freq = 5;
		std::string out_dir = "output\\";
		int reverse_history_size = 5;
		std::string video_file = "";
		BlockArray::Capture capture = BlockArray::Capture(NA_VALUE, NA_VALUE, NA_VALUE, BlockArray::Line::UP, BlockArray::CaptureType::CROSS);
		BlockArray::Line slit = BlockArray::Line(NA_VALUE, NA_VALUE, NA_VALUE, BlockArray::Line::UP);

		int search_radius = 1;
		double block_foreground_threshold = 0.5;

		bool interlayer_feedback = true;
		bool reverse_mrf = true;

		// Interlayer feedback
		double edge_threshold = 0.5;
		double edge_brightness_threshold = 0.1;
		double interval_threshold = 0.5;
		int min_edge_hamming_dist = 4;
	};
	// ALGO VARS
	Params params{};

	//Drawing
	const std::string WINDOW_NAME = "StMRF";
	int run_id = 0;
	bool ShowBlocksGrid = false;
	
#pragma region FORM

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Threading;
	/// <summary>
	/// —водка дл€ MainForm
	/// </summary>
	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm(void)
		{
			InitializeComponent();
		}	

	protected:
		/// <summary>
		/// ќсвободить все используемые ресурсы.
		/// </summary>

	private: System::Windows::Forms::TextBox^  ResultTextbox;
	private: System::Windows::Forms::GroupBox^  DataSourceGroup;
	private: System::Windows::Forms::TextBox^  SourcePathTextbox;



	private: System::Windows::Forms::OpenFileDialog^  VideoFileOpenDialog;
	private: System::Windows::Forms::Button^  CloseAppBtn;
	private: System::Windows::Forms::Button^  ChooseSourceFileOrFolderBtn;
	private: System::Windows::Forms::GroupBox^  VideoPlayerGroup;
	private: System::Windows::Forms::GroupBox^  InformationGroup;


	private: System::Windows::Forms::Button^  StartStopMainPlayerBtn;










	private: System::Windows::Forms::Label^  label4;
	private: System::Windows::Forms::CheckBox^  ShowBlocksGridCheckbox;











	private: System::Windows::Forms::Label^  label3;
	private: System::Windows::Forms::Label^  label5;
	private: System::Windows::Forms::TextBox^  SlitFromTextbox;

	private: System::Windows::Forms::Label^  label6;
	private: System::Windows::Forms::TextBox^  SlitToTextbox;

	private: System::Windows::Forms::RadioButton^  SlitsUpRadioBtn;
	private: System::Windows::Forms::RadioButton^  SlitsDownRadioBtn;

	private: System::Windows::Forms::Label^  label12;
	private: System::Windows::Forms::Label^  label1;
	private: System::Windows::Forms::TextBox^  SlitYTextbox;

	private: System::Windows::Forms::Label^  label10;
	private: System::Windows::Forms::Label^  label9;
	private: System::Windows::Forms::TextBox^  CaptureFromTextbox;

	private: System::Windows::Forms::Label^  label8;
	private: System::Windows::Forms::TextBox^  CaptureToTextbox;

	private: System::Windows::Forms::RadioButton^  CaptureCrossRadioBtn;
	private: System::Windows::Forms::RadioButton^  CaptureTouchRadioBtn;
	private: System::Windows::Forms::Label^  label7;
	private: System::Windows::Forms::TextBox^  CaptureYTextbox;

	private: System::Windows::Forms::Label^  label14;
	private: System::Windows::Forms::Label^  label13;
	private: System::Windows::Forms::TextBox^  ForegroundThTextbox;

	private: System::Windows::Forms::GroupBox^  SettingsGroup;
	private: System::Windows::Forms::TextBox^  ForegroundBlockThTextbox;

	private: System::Windows::Forms::Label^  label11;
	private: System::Windows::Forms::TextBox^  ReverseHistSizeTextbox;

	private: System::Windows::Forms::Label^  label15;
	private: System::Windows::Forms::TextBox^  FrameStepTextbox;

	private: System::Windows::Forms::Label^  label16;
	private: System::Windows::Forms::Label^  label17;
	private: System::Windows::Forms::TextBox^  SearchRadiusTextbox;

	private: System::Windows::Forms::Label^  label18;
	private: System::Windows::Forms::TextBox^  BlockSizeTextBox;
	private: System::Windows::Forms::TextBox^  FrameDelayTextbox;

	private: System::Windows::Forms::Label^  label19;
	private: System::Windows::Forms::Button^  CheckPositionsBtn;
	private: System::Windows::Forms::FolderBrowserDialog^  FolderBrowserDialog;
	private: System::Windows::Forms::GroupBox^  groupBox1;
	private: System::Windows::Forms::Button^  ChooseOutputDirBtn;
private: System::Windows::Forms::TextBox^  OutputDirTextbox;
private: System::Windows::Forms::CheckBox^  InterlayerFeedbackCheckbox;
private: System::Windows::Forms::CheckBox^  ReverseMrfCheckbox;
private: System::Windows::Forms::Label^  label2;
private: System::Windows::Forms::TextBox^  BackgroundFrameTextbox;

private: System::Windows::Forms::Label^  label20;
private: System::Windows::Forms::Panel^  panel1;

































	private: System::ComponentModel::IContainer^  components;

	protected:

	private:
		/// <summary>
		/// ќб€зательна€ переменна€ конструктора.
		/// </summary>

#pragma endregion

#pragma region Windows Form Designer generated code
		/// <summary>
		/// “ребуемый метод дл€ поддержки конструктора Ч не измен€йте 
		/// содержимое этого метода с помощью редактора кода.
		/// </summary>
		void InitializeComponent(void)
		{
			this->ResultTextbox = (gcnew System::Windows::Forms::TextBox());
			this->DataSourceGroup = (gcnew System::Windows::Forms::GroupBox());
			this->ChooseSourceFileOrFolderBtn = (gcnew System::Windows::Forms::Button());
			this->SourcePathTextbox = (gcnew System::Windows::Forms::TextBox());
			this->VideoFileOpenDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			this->CloseAppBtn = (gcnew System::Windows::Forms::Button());
			this->VideoPlayerGroup = (gcnew System::Windows::Forms::GroupBox());
			this->StartStopMainPlayerBtn = (gcnew System::Windows::Forms::Button());
			this->InformationGroup = (gcnew System::Windows::Forms::GroupBox());
			this->InterlayerFeedbackCheckbox = (gcnew System::Windows::Forms::CheckBox());
			this->ReverseMrfCheckbox = (gcnew System::Windows::Forms::CheckBox());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->ShowBlocksGridCheckbox = (gcnew System::Windows::Forms::CheckBox());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->label5 = (gcnew System::Windows::Forms::Label());
			this->SlitFromTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label6 = (gcnew System::Windows::Forms::Label());
			this->SlitToTextbox = (gcnew System::Windows::Forms::TextBox());
			this->SlitsUpRadioBtn = (gcnew System::Windows::Forms::RadioButton());
			this->SlitsDownRadioBtn = (gcnew System::Windows::Forms::RadioButton());
			this->label12 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->SlitYTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label10 = (gcnew System::Windows::Forms::Label());
			this->label9 = (gcnew System::Windows::Forms::Label());
			this->CaptureFromTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label8 = (gcnew System::Windows::Forms::Label());
			this->CaptureToTextbox = (gcnew System::Windows::Forms::TextBox());
			this->CaptureCrossRadioBtn = (gcnew System::Windows::Forms::RadioButton());
			this->CaptureTouchRadioBtn = (gcnew System::Windows::Forms::RadioButton());
			this->label7 = (gcnew System::Windows::Forms::Label());
			this->CaptureYTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label14 = (gcnew System::Windows::Forms::Label());
			this->label13 = (gcnew System::Windows::Forms::Label());
			this->ForegroundThTextbox = (gcnew System::Windows::Forms::TextBox());
			this->SettingsGroup = (gcnew System::Windows::Forms::GroupBox());
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->BackgroundFrameTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label20 = (gcnew System::Windows::Forms::Label());
			this->CheckPositionsBtn = (gcnew System::Windows::Forms::Button());
			this->FrameDelayTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label19 = (gcnew System::Windows::Forms::Label());
			this->BlockSizeTextBox = (gcnew System::Windows::Forms::TextBox());
			this->SearchRadiusTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label18 = (gcnew System::Windows::Forms::Label());
			this->ReverseHistSizeTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label15 = (gcnew System::Windows::Forms::Label());
			this->FrameStepTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label16 = (gcnew System::Windows::Forms::Label());
			this->label17 = (gcnew System::Windows::Forms::Label());
			this->ForegroundBlockThTextbox = (gcnew System::Windows::Forms::TextBox());
			this->label11 = (gcnew System::Windows::Forms::Label());
			this->FolderBrowserDialog = (gcnew System::Windows::Forms::FolderBrowserDialog());
			this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			this->ChooseOutputDirBtn = (gcnew System::Windows::Forms::Button());
			this->OutputDirTextbox = (gcnew System::Windows::Forms::TextBox());
			this->DataSourceGroup->SuspendLayout();
			this->VideoPlayerGroup->SuspendLayout();
			this->InformationGroup->SuspendLayout();
			this->SettingsGroup->SuspendLayout();
			this->panel1->SuspendLayout();
			this->groupBox1->SuspendLayout();
			this->SuspendLayout();
			// 
			// ResultTextbox
			// 
			this->ResultTextbox->BackColor = System::Drawing::SystemColors::ControlLight;
			this->ResultTextbox->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->ResultTextbox->Location = System::Drawing::Point(560, 12);
			this->ResultTextbox->Multiline = true;
			this->ResultTextbox->Name = L"ResultTextbox";
			this->ResultTextbox->ReadOnly = true;
			this->ResultTextbox->ScrollBars = System::Windows::Forms::ScrollBars::Vertical;
			this->ResultTextbox->Size = System::Drawing::Size(284, 398);
			this->ResultTextbox->TabIndex = 1;
			// 
			// DataSourceGroup
			// 
			this->DataSourceGroup->BackColor = System::Drawing::SystemColors::ControlLight;
			this->DataSourceGroup->Controls->Add(this->ChooseSourceFileOrFolderBtn);
			this->DataSourceGroup->Controls->Add(this->SourcePathTextbox);
			this->DataSourceGroup->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->DataSourceGroup->Location = System::Drawing::Point(12, 12);
			this->DataSourceGroup->Name = L"DataSourceGroup";
			this->DataSourceGroup->Size = System::Drawing::Size(251, 99);
			this->DataSourceGroup->TabIndex = 2;
			this->DataSourceGroup->TabStop = false;
			this->DataSourceGroup->Text = L"Data source";
			// 
			// ChooseSourceFileOrFolderBtn
			// 
			this->ChooseSourceFileOrFolderBtn->BackColor = System::Drawing::SystemColors::ButtonShadow;
			this->ChooseSourceFileOrFolderBtn->Location = System::Drawing::Point(6, 23);
			this->ChooseSourceFileOrFolderBtn->Name = L"ChooseSourceFileOrFolderBtn";
			this->ChooseSourceFileOrFolderBtn->Size = System::Drawing::Size(239, 33);
			this->ChooseSourceFileOrFolderBtn->TabIndex = 3;
			this->ChooseSourceFileOrFolderBtn->Text = L"Add path";
			this->ChooseSourceFileOrFolderBtn->UseVisualStyleBackColor = false;
			this->ChooseSourceFileOrFolderBtn->Click += gcnew System::EventHandler(this, &MainForm::ChooseSourceFileOrFolderBtn_Click);
			// 
			// SourcePathTextbox
			// 
			this->SourcePathTextbox->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->SourcePathTextbox->Location = System::Drawing::Point(6, 62);
			this->SourcePathTextbox->Name = L"SourcePathTextbox";
			this->SourcePathTextbox->ReadOnly = true;
			this->SourcePathTextbox->Size = System::Drawing::Size(239, 21);
			this->SourcePathTextbox->TabIndex = 2;
			// 
			// VideoFileOpenDialog
			// 
			this->VideoFileOpenDialog->FileName = L"VideoFileOpenDialog";
			this->VideoFileOpenDialog->Filter = L"Video files (*.mp4)|*.mp4";
			this->VideoFileOpenDialog->RestoreDirectory = true;
			// 
			// CloseAppBtn
			// 
			this->CloseAppBtn->BackColor = System::Drawing::SystemColors::ButtonShadow;
			this->CloseAppBtn->Location = System::Drawing::Point(6, 56);
			this->CloseAppBtn->Name = L"CloseAppBtn";
			this->CloseAppBtn->Size = System::Drawing::Size(111, 37);
			this->CloseAppBtn->TabIndex = 4;
			this->CloseAppBtn->Text = L"EXIT";
			this->CloseAppBtn->UseVisualStyleBackColor = false;
			this->CloseAppBtn->Click += gcnew System::EventHandler(this, &MainForm::CloseAppBtn_Click);
			// 
			// VideoPlayerGroup
			// 
			this->VideoPlayerGroup->BackColor = System::Drawing::SystemColors::ControlLight;
			this->VideoPlayerGroup->Controls->Add(this->CloseAppBtn);
			this->VideoPlayerGroup->Controls->Add(this->StartStopMainPlayerBtn);
			this->VideoPlayerGroup->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->VideoPlayerGroup->Location = System::Drawing::Point(389, 145);
			this->VideoPlayerGroup->Name = L"VideoPlayerGroup";
			this->VideoPlayerGroup->Size = System::Drawing::Size(136, 99);
			this->VideoPlayerGroup->TabIndex = 5;
			this->VideoPlayerGroup->TabStop = false;
			this->VideoPlayerGroup->Text = L"Video Player controls";
			// 
			// StartStopMainPlayerBtn
			// 
			this->StartStopMainPlayerBtn->BackColor = System::Drawing::SystemColors::ButtonShadow;
			this->StartStopMainPlayerBtn->Location = System::Drawing::Point(6, 21);
			this->StartStopMainPlayerBtn->Name = L"StartStopMainPlayerBtn";
			this->StartStopMainPlayerBtn->Size = System::Drawing::Size(111, 35);
			this->StartStopMainPlayerBtn->TabIndex = 0;
			this->StartStopMainPlayerBtn->Text = L"START";
			this->StartStopMainPlayerBtn->UseVisualStyleBackColor = false;
			this->StartStopMainPlayerBtn->Click += gcnew System::EventHandler(this, &MainForm::StartStopMainPlayerBtn_Click);
			// 
			// InformationGroup
			// 
			this->InformationGroup->Controls->Add(this->InterlayerFeedbackCheckbox);
			this->InformationGroup->Controls->Add(this->ReverseMrfCheckbox);
			this->InformationGroup->Controls->Add(this->label2);
			this->InformationGroup->Controls->Add(this->ShowBlocksGridCheckbox);
			this->InformationGroup->Controls->Add(this->label4);
			this->InformationGroup->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->InformationGroup->Location = System::Drawing::Point(389, 267);
			this->InformationGroup->Name = L"InformationGroup";
			this->InformationGroup->Size = System::Drawing::Size(146, 122);
			this->InformationGroup->TabIndex = 5;
			this->InformationGroup->TabStop = false;
			this->InformationGroup->Text = L"Settings";
			// 
			// InterlayerFeedbackCheckbox
			// 
			this->InterlayerFeedbackCheckbox->AutoSize = true;
			this->InterlayerFeedbackCheckbox->Location = System::Drawing::Point(9, 92);
			this->InterlayerFeedbackCheckbox->Name = L"InterlayerFeedbackCheckbox";
			this->InterlayerFeedbackCheckbox->Size = System::Drawing::Size(133, 19);
			this->InterlayerFeedbackCheckbox->TabIndex = 13;
			this->InterlayerFeedbackCheckbox->Text = L"Interlayer Feedback";
			this->InterlayerFeedbackCheckbox->UseVisualStyleBackColor = true;
			// 
			// ReverseMrfCheckbox
			// 
			this->ReverseMrfCheckbox->AutoSize = true;
			this->ReverseMrfCheckbox->Location = System::Drawing::Point(9, 69);
			this->ReverseMrfCheckbox->Name = L"ReverseMrfCheckbox";
			this->ReverseMrfCheckbox->Size = System::Drawing::Size(120, 19);
			this->ReverseMrfCheckbox->TabIndex = 11;
			this->ReverseMrfCheckbox->Text = L"Reverse ST-MRF";
			this->ReverseMrfCheckbox->UseVisualStyleBackColor = true;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label2->Location = System::Drawing::Point(6, 48);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(49, 15);
			this->label2->TabIndex = 12;
			this->label2->Text = L"Enable:";
			// 
			// ShowBlocksGridCheckbox
			// 
			this->ShowBlocksGridCheckbox->AutoSize = true;
			this->ShowBlocksGridCheckbox->Location = System::Drawing::Point(53, 24);
			this->ShowBlocksGridCheckbox->Name = L"ShowBlocksGridCheckbox";
			this->ShowBlocksGridCheckbox->Size = System::Drawing::Size(86, 19);
			this->ShowBlocksGridCheckbox->TabIndex = 9;
			this->ShowBlocksGridCheckbox->Text = L"Blocks grid";
			this->ShowBlocksGridCheckbox->UseVisualStyleBackColor = true;
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label4->Location = System::Drawing::Point(6, 25);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(41, 15);
			this->label4->TabIndex = 10;
			this->label4->Text = L"Show:";
			// 
			// label3
			// 
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label3->Location = System::Drawing::Point(6, 20);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(78, 15);
			this->label3->TabIndex = 7;
			this->label3->Text = L"Slits params:";
			// 
			// label5
			// 
			this->label5->AutoSize = true;
			this->label5->Location = System::Drawing::Point(65, 40);
			this->label5->Name = L"label5";
			this->label5->Size = System::Drawing::Size(36, 15);
			this->label5->TabIndex = 11;
			this->label5->Text = L"From";
			// 
			// SlitFromTextbox
			// 
			this->SlitFromTextbox->Location = System::Drawing::Point(107, 37);
			this->SlitFromTextbox->Name = L"SlitFromTextbox";
			this->SlitFromTextbox->Size = System::Drawing::Size(45, 21);
			this->SlitFromTextbox->TabIndex = 12;
			this->SlitFromTextbox->Text = L"0";
			this->SlitFromTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label6
			// 
			this->label6->AutoSize = true;
			this->label6->Location = System::Drawing::Point(80, 61);
			this->label6->Name = L"label6";
			this->label6->Size = System::Drawing::Size(21, 15);
			this->label6->TabIndex = 13;
			this->label6->Text = L"To";
			// 
			// SlitToTextbox
			// 
			this->SlitToTextbox->Location = System::Drawing::Point(107, 58);
			this->SlitToTextbox->Name = L"SlitToTextbox";
			this->SlitToTextbox->Size = System::Drawing::Size(45, 21);
			this->SlitToTextbox->TabIndex = 14;
			this->SlitToTextbox->Text = L"600";
			this->SlitToTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// SlitsUpRadioBtn
			// 
			this->SlitsUpRadioBtn->AutoSize = true;
			this->SlitsUpRadioBtn->Location = System::Drawing::Point(0, 3);
			this->SlitsUpRadioBtn->Name = L"SlitsUpRadioBtn";
			this->SlitsUpRadioBtn->Size = System::Drawing::Size(41, 19);
			this->SlitsUpRadioBtn->TabIndex = 18;
			this->SlitsUpRadioBtn->Text = L"Up";
			this->SlitsUpRadioBtn->UseVisualStyleBackColor = true;
			// 
			// SlitsDownRadioBtn
			// 
			this->SlitsDownRadioBtn->AutoSize = true;
			this->SlitsDownRadioBtn->Checked = true;
			this->SlitsDownRadioBtn->Location = System::Drawing::Point(0, 23);
			this->SlitsDownRadioBtn->Name = L"SlitsDownRadioBtn";
			this->SlitsDownRadioBtn->Size = System::Drawing::Size(57, 19);
			this->SlitsDownRadioBtn->TabIndex = 19;
			this->SlitsDownRadioBtn->TabStop = true;
			this->SlitsDownRadioBtn->Text = L"Down";
			this->SlitsDownRadioBtn->UseVisualStyleBackColor = true;
			// 
			// label12
			// 
			this->label12->AutoSize = true;
			this->label12->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label12->Location = System::Drawing::Point(7, 208);
			this->label12->Name = L"label12";
			this->label12->Size = System::Drawing::Size(65, 15);
			this->label12->TabIndex = 26;
			this->label12->Text = L"Block size:";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Location = System::Drawing::Point(89, 83);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(12, 15);
			this->label1->TabIndex = 27;
			this->label1->Text = L"y";
			// 
			// SlitYTextbox
			// 
			this->SlitYTextbox->Location = System::Drawing::Point(107, 80);
			this->SlitYTextbox->Name = L"SlitYTextbox";
			this->SlitYTextbox->Size = System::Drawing::Size(45, 21);
			this->SlitYTextbox->TabIndex = 28;
			this->SlitYTextbox->Text = L"50";
			this->SlitYTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label10
			// 
			this->label10->AutoSize = true;
			this->label10->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label10->Location = System::Drawing::Point(6, 115);
			this->label10->Name = L"label10";
			this->label10->Size = System::Drawing::Size(98, 15);
			this->label10->TabIndex = 29;
			this->label10->Text = L"Capture params:";
			// 
			// label9
			// 
			this->label9->AutoSize = true;
			this->label9->Location = System::Drawing::Point(65, 135);
			this->label9->Name = L"label9";
			this->label9->Size = System::Drawing::Size(36, 15);
			this->label9->TabIndex = 30;
			this->label9->Text = L"From";
			// 
			// CaptureFromTextbox
			// 
			this->CaptureFromTextbox->Location = System::Drawing::Point(107, 132);
			this->CaptureFromTextbox->Name = L"CaptureFromTextbox";
			this->CaptureFromTextbox->Size = System::Drawing::Size(45, 21);
			this->CaptureFromTextbox->TabIndex = 31;
			this->CaptureFromTextbox->Text = L"0";
			this->CaptureFromTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label8
			// 
			this->label8->AutoSize = true;
			this->label8->Location = System::Drawing::Point(80, 156);
			this->label8->Name = L"label8";
			this->label8->Size = System::Drawing::Size(21, 15);
			this->label8->TabIndex = 32;
			this->label8->Text = L"To";
			// 
			// CaptureToTextbox
			// 
			this->CaptureToTextbox->Location = System::Drawing::Point(107, 153);
			this->CaptureToTextbox->Name = L"CaptureToTextbox";
			this->CaptureToTextbox->Size = System::Drawing::Size(45, 21);
			this->CaptureToTextbox->TabIndex = 33;
			this->CaptureToTextbox->Text = L"600";
			this->CaptureToTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// CaptureCrossRadioBtn
			// 
			this->CaptureCrossRadioBtn->AutoSize = true;
			this->CaptureCrossRadioBtn->Checked = true;
			this->CaptureCrossRadioBtn->Location = System::Drawing::Point(10, 133);
			this->CaptureCrossRadioBtn->Name = L"CaptureCrossRadioBtn";
			this->CaptureCrossRadioBtn->Size = System::Drawing::Size(56, 19);
			this->CaptureCrossRadioBtn->TabIndex = 34;
			this->CaptureCrossRadioBtn->TabStop = true;
			this->CaptureCrossRadioBtn->Text = L"Cross";
			this->CaptureCrossRadioBtn->UseVisualStyleBackColor = true;
			// 
			// CaptureTouchRadioBtn
			// 
			this->CaptureTouchRadioBtn->AutoSize = true;
			this->CaptureTouchRadioBtn->Location = System::Drawing::Point(10, 154);
			this->CaptureTouchRadioBtn->Name = L"CaptureTouchRadioBtn";
			this->CaptureTouchRadioBtn->Size = System::Drawing::Size(59, 19);
			this->CaptureTouchRadioBtn->TabIndex = 35;
			this->CaptureTouchRadioBtn->Text = L"Touch";
			this->CaptureTouchRadioBtn->UseVisualStyleBackColor = true;
			// 
			// label7
			// 
			this->label7->AutoSize = true;
			this->label7->Location = System::Drawing::Point(89, 178);
			this->label7->Name = L"label7";
			this->label7->Size = System::Drawing::Size(12, 15);
			this->label7->TabIndex = 36;
			this->label7->Text = L"y";
			// 
			// CaptureYTextbox
			// 
			this->CaptureYTextbox->Location = System::Drawing::Point(107, 175);
			this->CaptureYTextbox->Name = L"CaptureYTextbox";
			this->CaptureYTextbox->Size = System::Drawing::Size(45, 21);
			this->CaptureYTextbox->TabIndex = 37;
			this->CaptureYTextbox->Text = L"350";
			this->CaptureYTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label14
			// 
			this->label14->AutoSize = true;
			this->label14->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label14->Location = System::Drawing::Point(186, 16);
			this->label14->Name = L"label14";
			this->label14->Size = System::Drawing::Size(107, 15);
			this->label14->TabIndex = 38;
			this->label14->Text = L"Algorithm params:";
			// 
			// label13
			// 
			this->label13->AutoSize = true;
			this->label13->Location = System::Drawing::Point(210, 37);
			this->label13->Name = L"label13";
			this->label13->Size = System::Drawing::Size(71, 30);
			this->label13->TabIndex = 39;
			this->label13->Text = L"Foreground\r\nthreshold";
			this->label13->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// ForegroundThTextbox
			// 
			this->ForegroundThTextbox->Location = System::Drawing::Point(287, 42);
			this->ForegroundThTextbox->Name = L"ForegroundThTextbox";
			this->ForegroundThTextbox->Size = System::Drawing::Size(45, 21);
			this->ForegroundThTextbox->TabIndex = 40;
			this->ForegroundThTextbox->Text = L"0.1";
			this->ForegroundThTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// SettingsGroup
			// 
			this->SettingsGroup->Controls->Add(this->panel1);
			this->SettingsGroup->Controls->Add(this->BackgroundFrameTextbox);
			this->SettingsGroup->Controls->Add(this->label20);
			this->SettingsGroup->Controls->Add(this->CheckPositionsBtn);
			this->SettingsGroup->Controls->Add(this->FrameDelayTextbox);
			this->SettingsGroup->Controls->Add(this->label19);
			this->SettingsGroup->Controls->Add(this->BlockSizeTextBox);
			this->SettingsGroup->Controls->Add(this->SearchRadiusTextbox);
			this->SettingsGroup->Controls->Add(this->label18);
			this->SettingsGroup->Controls->Add(this->ReverseHistSizeTextbox);
			this->SettingsGroup->Controls->Add(this->label15);
			this->SettingsGroup->Controls->Add(this->FrameStepTextbox);
			this->SettingsGroup->Controls->Add(this->label16);
			this->SettingsGroup->Controls->Add(this->label17);
			this->SettingsGroup->Controls->Add(this->ForegroundBlockThTextbox);
			this->SettingsGroup->Controls->Add(this->label11);
			this->SettingsGroup->Controls->Add(this->ForegroundThTextbox);
			this->SettingsGroup->Controls->Add(this->label13);
			this->SettingsGroup->Controls->Add(this->label14);
			this->SettingsGroup->Controls->Add(this->CaptureYTextbox);
			this->SettingsGroup->Controls->Add(this->label7);
			this->SettingsGroup->Controls->Add(this->CaptureTouchRadioBtn);
			this->SettingsGroup->Controls->Add(this->CaptureCrossRadioBtn);
			this->SettingsGroup->Controls->Add(this->CaptureToTextbox);
			this->SettingsGroup->Controls->Add(this->label8);
			this->SettingsGroup->Controls->Add(this->CaptureFromTextbox);
			this->SettingsGroup->Controls->Add(this->label9);
			this->SettingsGroup->Controls->Add(this->label10);
			this->SettingsGroup->Controls->Add(this->SlitYTextbox);
			this->SettingsGroup->Controls->Add(this->label1);
			this->SettingsGroup->Controls->Add(this->label12);
			this->SettingsGroup->Controls->Add(this->SlitToTextbox);
			this->SettingsGroup->Controls->Add(this->label6);
			this->SettingsGroup->Controls->Add(this->SlitFromTextbox);
			this->SettingsGroup->Controls->Add(this->label5);
			this->SettingsGroup->Controls->Add(this->label3);
			this->SettingsGroup->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->SettingsGroup->Location = System::Drawing::Point(12, 129);
			this->SettingsGroup->Name = L"SettingsGroup";
			this->SettingsGroup->Size = System::Drawing::Size(353, 288);
			this->SettingsGroup->TabIndex = 6;
			this->SettingsGroup->TabStop = false;
			this->SettingsGroup->Text = L"Settings";
			// 
			// panel1
			// 
			this->panel1->Controls->Add(this->SlitsUpRadioBtn);
			this->panel1->Controls->Add(this->SlitsDownRadioBtn);
			this->panel1->Location = System::Drawing::Point(9, 38);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(57, 51);
			this->panel1->TabIndex = 55;
			// 
			// BackgroundFrameTextbox
			// 
			this->BackgroundFrameTextbox->Location = System::Drawing::Point(286, 239);
			this->BackgroundFrameTextbox->Name = L"BackgroundFrameTextbox";
			this->BackgroundFrameTextbox->Size = System::Drawing::Size(45, 21);
			this->BackgroundFrameTextbox->TabIndex = 54;
			this->BackgroundFrameTextbox->Text = L"100";
			this->BackgroundFrameTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label20
			// 
			this->label20->AutoSize = true;
			this->label20->Location = System::Drawing::Point(190, 233);
			this->label20->Name = L"label20";
			this->label20->Size = System::Drawing::Size(91, 30);
			this->label20->TabIndex = 53;
			this->label20->Text = L"#Frames for\r\nbackground init";
			this->label20->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// CheckPositionsBtn
			// 
			this->CheckPositionsBtn->BackColor = System::Drawing::SystemColors::ButtonShadow;
			this->CheckPositionsBtn->Location = System::Drawing::Point(27, 235);
			this->CheckPositionsBtn->Name = L"CheckPositionsBtn";
			this->CheckPositionsBtn->Size = System::Drawing::Size(111, 35);
			this->CheckPositionsBtn->TabIndex = 5;
			this->CheckPositionsBtn->Text = L"CHECK";
			this->CheckPositionsBtn->UseVisualStyleBackColor = false;
			this->CheckPositionsBtn->Click += gcnew System::EventHandler(this, &MainForm::CheckPositionsBtn_Click);
			// 
			// FrameDelayTextbox
			// 
			this->FrameDelayTextbox->Location = System::Drawing::Point(286, 158);
			this->FrameDelayTextbox->Name = L"FrameDelayTextbox";
			this->FrameDelayTextbox->Size = System::Drawing::Size(45, 21);
			this->FrameDelayTextbox->TabIndex = 52;
			this->FrameDelayTextbox->Text = L"30";
			this->FrameDelayTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label19
			// 
			this->label19->AutoSize = true;
			this->label19->Location = System::Drawing::Point(204, 159);
			this->label19->Name = L"label19";
			this->label19->Size = System::Drawing::Size(75, 15);
			this->label19->TabIndex = 51;
			this->label19->Text = L"Frame delay";
			this->label19->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// BlockSizeTextBox
			// 
			this->BlockSizeTextBox->Location = System::Drawing::Point(107, 208);
			this->BlockSizeTextBox->Name = L"BlockSizeTextBox";
			this->BlockSizeTextBox->Size = System::Drawing::Size(45, 21);
			this->BlockSizeTextBox->TabIndex = 50;
			this->BlockSizeTextBox->Text = L"20";
			this->BlockSizeTextBox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// SearchRadiusTextbox
			// 
			this->SearchRadiusTextbox->Location = System::Drawing::Point(286, 212);
			this->SearchRadiusTextbox->Name = L"SearchRadiusTextbox";
			this->SearchRadiusTextbox->Size = System::Drawing::Size(45, 21);
			this->SearchRadiusTextbox->TabIndex = 49;
			this->SearchRadiusTextbox->Text = L"5";
			this->SearchRadiusTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label18
			// 
			this->label18->AutoSize = true;
			this->label18->Location = System::Drawing::Point(199, 215);
			this->label18->Name = L"label18";
			this->label18->Size = System::Drawing::Size(83, 15);
			this->label18->TabIndex = 48;
			this->label18->Text = L"Search radius";
			this->label18->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// ReverseHistSizeTextbox
			// 
			this->ReverseHistSizeTextbox->Location = System::Drawing::Point(286, 186);
			this->ReverseHistSizeTextbox->Name = L"ReverseHistSizeTextbox";
			this->ReverseHistSizeTextbox->Size = System::Drawing::Size(45, 21);
			this->ReverseHistSizeTextbox->TabIndex = 47;
			this->ReverseHistSizeTextbox->Text = L"3";
			this->ReverseHistSizeTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label15
			// 
			this->label15->AutoSize = true;
			this->label15->Location = System::Drawing::Point(212, 181);
			this->label15->Name = L"label15";
			this->label15->Size = System::Drawing::Size(67, 30);
			this->label15->TabIndex = 46;
			this->label15->Text = L"Reverse\r\nhistory size";
			this->label15->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// FrameStepTextbox
			// 
			this->FrameStepTextbox->Location = System::Drawing::Point(286, 134);
			this->FrameStepTextbox->Name = L"FrameStepTextbox";
			this->FrameStepTextbox->Size = System::Drawing::Size(45, 21);
			this->FrameStepTextbox->TabIndex = 45;
			this->FrameStepTextbox->Text = L"2";
			this->FrameStepTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label16
			// 
			this->label16->AutoSize = true;
			this->label16->Location = System::Drawing::Point(210, 135);
			this->label16->Name = L"label16";
			this->label16->Size = System::Drawing::Size(69, 15);
			this->label16->TabIndex = 44;
			this->label16->Text = L"Frame step";
			this->label16->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// label17
			// 
			this->label17->AutoSize = true;
			this->label17->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Underline, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->label17->Location = System::Drawing::Point(191, 116);
			this->label17->Name = L"label17";
			this->label17->Size = System::Drawing::Size(126, 15);
			this->label17->TabIndex = 43;
			this->label17->Text = L"Performance params:";
			// 
			// ForegroundBlockThTextbox
			// 
			this->ForegroundBlockThTextbox->Location = System::Drawing::Point(287, 80);
			this->ForegroundBlockThTextbox->Name = L"ForegroundBlockThTextbox";
			this->ForegroundBlockThTextbox->Size = System::Drawing::Size(45, 21);
			this->ForegroundBlockThTextbox->TabIndex = 42;
			this->ForegroundBlockThTextbox->Text = L"0.5";
			this->ForegroundBlockThTextbox->TextAlign = System::Windows::Forms::HorizontalAlignment::Center;
			// 
			// label11
			// 
			this->label11->AutoSize = true;
			this->label11->Location = System::Drawing::Point(191, 74);
			this->label11->Name = L"label11";
			this->label11->Size = System::Drawing::Size(90, 30);
			this->label11->TabIndex = 41;
			this->label11->Text = L"Foreground\r\nblock threshold";
			this->label11->TextAlign = System::Drawing::ContentAlignment::MiddleRight;
			// 
			// groupBox1
			// 
			this->groupBox1->BackColor = System::Drawing::SystemColors::ControlLight;
			this->groupBox1->Controls->Add(this->ChooseOutputDirBtn);
			this->groupBox1->Controls->Add(this->OutputDirTextbox);
			this->groupBox1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(204)));
			this->groupBox1->Location = System::Drawing::Point(284, 12);
			this->groupBox1->Name = L"groupBox1";
			this->groupBox1->Size = System::Drawing::Size(251, 99);
			this->groupBox1->TabIndex = 4;
			this->groupBox1->TabStop = false;
			this->groupBox1->Text = L"Output path";
			// 
			// ChooseOutputDirBtn
			// 
			this->ChooseOutputDirBtn->BackColor = System::Drawing::SystemColors::ButtonShadow;
			this->ChooseOutputDirBtn->Location = System::Drawing::Point(6, 23);
			this->ChooseOutputDirBtn->Name = L"ChooseOutputDirBtn";
			this->ChooseOutputDirBtn->Size = System::Drawing::Size(239, 33);
			this->ChooseOutputDirBtn->TabIndex = 3;
			this->ChooseOutputDirBtn->Text = L"Add path";
			this->ChooseOutputDirBtn->UseVisualStyleBackColor = false;
			this->ChooseOutputDirBtn->Click += gcnew System::EventHandler(this, &MainForm::ChooseOutputDirBtn_Click);
			// 
			// OutputDirTextbox
			// 
			this->OutputDirTextbox->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->OutputDirTextbox->Location = System::Drawing::Point(6, 62);
			this->OutputDirTextbox->Name = L"OutputDirTextbox";
			this->OutputDirTextbox->ReadOnly = true;
			this->OutputDirTextbox->Size = System::Drawing::Size(239, 21);
			this->OutputDirTextbox->TabIndex = 2;
			// 
			// MainForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(856, 439);
			this->ControlBox = false;
			this->Controls->Add(this->groupBox1);
			this->Controls->Add(this->SettingsGroup);
			this->Controls->Add(this->InformationGroup);
			this->Controls->Add(this->VideoPlayerGroup);
			this->Controls->Add(this->DataSourceGroup);
			this->Controls->Add(this->ResultTextbox);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedSingle;
			this->Name = L"MainForm";
			this->Text = L"Vehicles Tracker";
			this->DataSourceGroup->ResumeLayout(false);
			this->DataSourceGroup->PerformLayout();
			this->VideoPlayerGroup->ResumeLayout(false);
			this->InformationGroup->ResumeLayout(false);
			this->InformationGroup->PerformLayout();
			this->SettingsGroup->ResumeLayout(false);
			this->SettingsGroup->PerformLayout();
			this->panel1->ResumeLayout(false);
			this->panel1->PerformLayout();
			this->groupBox1->ResumeLayout(false);
			this->groupBox1->PerformLayout();
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion
	
	bool isStartAutoPlayer = false;
	
	private: System::Void CloseAppBtn_Click(System::Object^  sender, System::EventArgs^  e) 
	{
		if (bVideoSourceAnalyzerThreadState)
		{
			bVideoSourceAnalyzerThreadState = false;
			this->tVideoSourceAnalyzerThread->Abort();
		}
		Sleep(100);
		Application::Exit();
	}

	private: System::Void ChooseSourceFileOrFolderBtn_Click(System::Object^  sender, System::EventArgs^  e)
	{		
		String^ _sSourcePath = L"";
		if (VideoFileOpenDialog->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			_sSourcePath = VideoFileOpenDialog->FileName;
		}
		SourcePathTextbox->Text = _sSourcePath;
	}

	private: System::Void ChooseOutputDirBtn_Click(System::Object^  sender, System::EventArgs^  e)
	{
		String^ _sSourcePath = L"";
		if (FolderBrowserDialog->ShowDialog() == System::Windows::Forms::DialogResult::OK)
		{
			_sSourcePath = FolderBrowserDialog->SelectedPath;
		}
		OutputDirTextbox->Text = _sSourcePath;
	}

	private: System::Void StartStopMainPlayerBtn_Click(System::Object^  sender, System::EventArgs^  e) 
	{
		StartStopMainPlayerBtn_ClickEvent();
	}

	private: System::Void CheckPositionsBtn_Click(System::Object^  sender, System::EventArgs^  e) 
	{
		CheckPositionsBtn_ClickEvent();
	}
	
	#pragma region MainActions
	private: void CheckPositionsBtn_ClickEvent()
	{
		const std::string wind_name = "First frame";
		const int thickness = 2;

		if (SourcePathTextbox->Text == "")
		{
			ResultTextbox->Text += "Error: Path to source should be choosen\r\n";
			isStartAutoPlayer = !isStartAutoPlayer;
			return;
		}

		Params p = ParseParams();
		std::string sSourcePath = msclr::interop::marshal_as<std::string>(SourcePathTextbox->Text);
		auto cCap = cv::VideoCapture(sSourcePath);
		cv::Mat frame;

		if (!cCap.isOpened() || !read_frame(cCap, frame))
		{
			this->Trace("Can't open video file\r\n");
			return;
		}

		cv::line(frame, cv::Point(p.slit.x_left, p.slit.y), cv::Point(p.slit.x_right, p.slit.y), CV_RGB(1, 0, 0), thickness);
		cv::line(frame, cv::Point(p.slit.x_left, p.slit.y + p.block_height), cv::Point(p.slit.x_right, p.slit.y + p.block_height), CV_RGB(1, 0, 0), thickness);

		cv::line(frame, cv::Point(p.capture.x_left, p.capture.y), cv::Point(p.capture.x_right, p.capture.y), CV_RGB(0, 0, 1), thickness);

		show_image(frame, wind_name, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		cv::destroyWindow(wind_name);
	}

	private: Params ParseParams()
	{
		Params p;
		p.slit.x_left = Int32::Parse(SlitFromTextbox->Text);
		p.slit.x_right = Int32::Parse(SlitToTextbox->Text);
		p.slit.y = Int32::Parse(SlitYTextbox->Text);

		if (SlitsUpRadioBtn->Checked)
		{
			p.slit.direction = BlockArray::Line::UP;
			p.capture.direction = BlockArray::Line::UP;
		}
		else
		{
			p.slit.direction = BlockArray::Line::DOWN;
			p.capture.direction = BlockArray::Line::DOWN;
		}

		p.capture.x_left = Int32::Parse(CaptureFromTextbox->Text);
		p.capture.x_right = Int32::Parse(CaptureToTextbox->Text);
		p.capture.y = Int32::Parse(CaptureYTextbox->Text);

		if (CaptureCrossRadioBtn->Checked)
		{
			p.capture.type = BlockArray::CaptureType::CROSS;
		}
		else
		{
			p.capture.type = BlockArray::CaptureType::TOUCH;
		}

		auto culture_info = System::Globalization::CultureInfo::GetCultureInfo("en-US");
		p.foreground_threshold = Double::Parse(ForegroundThTextbox->Text, culture_info);
		p.block_foreground_threshold = Double::Parse(ForegroundBlockThTextbox->Text, culture_info);

		p.background_init_length = Int32::Parse(BackgroundFrameTextbox->Text);
		p.frame_freq = Int32::Parse(FrameStepTextbox->Text);
		p.frame_delay = Int32::Parse(FrameDelayTextbox->Text);
		p.reverse_history_size = Int32::Parse(ReverseHistSizeTextbox->Text);
		p.search_radius = Int32::Parse(SearchRadiusTextbox->Text);

		p.block_width = Int32::Parse(BlockSizeTextBox->Text);
		p.block_height = p.block_width;
		
		p.out_dir = msclr::interop::marshal_as<std::string>(OutputDirTextbox->Text);

		p.reverse_mrf = ReverseMrfCheckbox->Checked;
		p.interlayer_feedback = InterlayerFeedbackCheckbox->Checked;

		if (!p.reverse_mrf)
		{
			p.reverse_history_size = 0;
		}

		return p;
	}

	private: void StartStopMainPlayerBtn_ClickEvent()
	{
		isStartAutoPlayer = !isStartAutoPlayer;

		if (isStartAutoPlayer) //player started
		{
			if (SourcePathTextbox->Text == "")
			{
				ResultTextbox->Text += "Error: Path to source should be choosen\r\nMain player stopped\r\n";
				isStartAutoPlayer = !isStartAutoPlayer;
				return;
			}

			if (OutputDirTextbox->Text == "")
			{
				ResultTextbox->Text += "Error: Output durectory should be choosen\r\nMain player stopped\r\n";
				isStartAutoPlayer = !isStartAutoPlayer;
				return;
			}
			
			StartStopMainPlayerBtn->Text = "STOP";
			DataSourceGroup->Enabled = false;
			SettingsGroup->Enabled = false;

			#pragma region READ_VALUES_FROM_VIEW
			//INIT!
			params = ParseParams();
			
			ShowBlocksGrid = ShowBlocksGridCheckbox->Checked;

			String ^out_dir = gcnew String(params.out_dir.c_str());
			if (System::IO::Directory::Exists(out_dir)) {
				System::IO::Directory::CreateDirectory(out_dir);
			}
			#pragma endregion

			//Start video analyzer thread and set path
			bVideoSourceAnalyzerThreadState = true;
			this->tVideoSourceAnalyzerThread = gcnew Thread(gcnew System::Threading::ParameterizedThreadStart(this, &MainForm::VideoSourceAnalyzer));
			this->tVideoSourceAnalyzerThread->Start(SourcePathTextbox->Text);
			ResultTextbox->Text += "Video player initialized\r\n";
		}
		else //player stopped
		{
			StartStopMainPlayerBtn->Text = "START";
			DataSourceGroup->Enabled = true;
			SettingsGroup->Enabled = true;
			
			if (bVideoSourceAnalyzerThreadState)
			{
				bVideoSourceAnalyzerThreadState = false;
				this->tVideoSourceAnalyzerThread->Abort();

				ResultTextbox->Text += "Video player stopped\r\n";
			}

			cv::destroyAllWindows();
		}
	}

	#pragma endregion

	#pragma region General
	bool bVideoSourceAnalyzerThreadState = false;
	Thread^ tVideoSourceAnalyzerThread;
	delegate void MyInvokeDelegate(String^ msg);

	private: void AddMessageToResultTextbox(String^ msg)
	{
		ResultTextbox->Text += msg;
	}

	private: void Trace(String^ msg)
	{
		this->BeginInvoke(gcnew MyInvokeDelegate(this, &MainForm::AddMessageToResultTextbox), msg);
	}
	#pragma endregion

	Tracker get_tracker(const Params &p, const cv::Mat &background)
	{
		BlockArray blocks(background.rows / p.block_height, background.cols / p.block_width, p.block_height, p.block_width);
		if (p.slit.y > background.rows - background.rows % p.block_height)
			throw std::logic_error("Slit y is too large: " + std::to_string(p.slit.y));

		BlockArray::Slit slit(p.slit, p.block_width, p.block_height);

		return Tracker(p.foreground_threshold, p.background_update_weight, p.reverse_history_size,
			p.search_radius, p.block_foreground_threshold, p.interlayer_feedback, p.edge_threshold, p.edge_brightness_threshold,
			p.interval_threshold, p.min_edge_hamming_dist, background, slit, p.capture, blocks);
	}

	#pragma region MainAlgo
	private: void VideoSourceAnalyzer(Object^ path)
	{
		srand(time(0));

		String^ sourcePath = (String^)path;
		std::string sSourcePath = msclr::interop::marshal_as<std::string>(sourcePath);
		TextBox^ tbResultsTextbox = ResultTextbox;

		//BACKGROUND ACCUMULATION!
		this->Trace("step (0): Background accumulation starting..\r\n");
		auto background = estimate_background(sSourcePath, params.background_init_length, params.background_update_weight, 3);
		/*cv::Mat background;
		if (System::IO::File::Exists("background.jpg"))
		{
			background = cv::imread("background.jpg");
			background.convertTo(background, CV_32FC3, 1.0 / 255.0);
		}
		else
		{
			background = estimate_background(sSourcePath, params.background_init_length, params.background_update_weight, 3);
			cv::Mat background_out = background.clone() * 255;
			background_out.convertTo(background_out, CV_8UC3);
			cv::imwrite("background.jpg", background_out);
		}*/
		
		this->Trace("step (0): Done\r\n");

		//MAIN ALGO INIT!
		this->Trace("step (1): Tracker initializing..\r\n");

		int nFrameCounter = 0;
		auto cCap = cv::VideoCapture(sSourcePath);

		cv::Mat frame;
		if (!cCap.isOpened() || !read_frame(cCap, frame))
		{
			this->Trace("Can't open video file\r\n");
			return;
		}
		size_t out_id = 0;
		cv::Mat old_frame;
		Tracker tracker = get_tracker(params, background);
		tracker.add_frame(frame);

		this->Trace("step (1): Done\r\nTracker launched successfully\r\n");
		//Start tracker!
		run_id++;
		cv::namedWindow(WINDOW_NAME + std::to_string(run_id), CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		try
		{
			while (read_frame(cCap, frame))
			{
				auto start = std::chrono::steady_clock::now();

				nFrameCounter++;
				//capture next frame

				tracker.add_frame(frame);
				// auto reg_vehicle_ids = tracker.register_vehicle_step(frame, old_frame, background);
				id_set_t reg_vehicle_ids;
				if (params.reverse_mrf)
				{
					reg_vehicle_ids = tracker.reverse_st_mrf_step();
				}
				else
				{
					reg_vehicle_ids = tracker.register_vehicle_step(frame, old_frame, tracker.background());
				}
				auto b_boxes = bounding_boxes(tracker.blocks());
				for (auto id : reg_vehicle_ids)
				{
					save_vehicle(frame, b_boxes.at(id), params.out_dir, out_id++);
				}

				old_frame = frame;
				auto plt_frame = plot_frame(frame, tracker.blocks(), tracker.slit, tracker.capture);
				if (ShowBlocksGrid)
				{
					draw_grid(plt_frame, tracker.blocks());
				}
				//Refresh image in frame box
				cv::imshow(WINDOW_NAME + std::to_string(run_id), plt_frame);

				auto elapsed = std::chrono::steady_clock::now() - start;
				int spentTime = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
				cv::waitKey(std::max(params.frame_delay - spentTime, 1));
			}
		}
		catch (const std::runtime_error &err)
		{
			std::string message = std::string("ERROR: ") + err.what();
			this->Trace(gcnew String(message.c_str()));
		}

		cv::destroyWindow(WINDOW_NAME + std::to_string(run_id));
	}
	#pragma endregion
};
}
