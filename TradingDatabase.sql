/****** Object:  Table [dbo].[HourData]    Script Date: 11/29/2021 4:31:25 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE TABLE [dbo].[HourData](
	[DateIndex] [smalldatetime] NULL,
	[High] [float] NULL,
	[Low] [float] NULL,
	[Close_] [float] NULL,
	[Volume] [int] NULL,
	[Open_] [float] NULL,
	[Ticker] [varchar](5) NULL,
	[Id] [int] IDENTITY(1,1) NOT NULL
) ON [PRIMARY]
GO

CREATE TABLE [dbo].[Tickers](
	[Symbol] [varchar](5) NULL,
	[Rank] [float] NULL,
	[Trained_Date] [date] NULL,
	[Predicted_Inc] [float] NULL,
	[Model_Accuracy] [decimal](18, 5) NULL,
	[TrainingParams] [varchar](max) NULL,
	[trained_filename] [varchar](100) NULL,
	[CompanyName] [varchar](200) NULL,
	[Sector] [varchar](200) NULL,
	[Industry] [varchar](200) NULL,
	[IndexCol] [int] IDENTITY(1,1) NOT NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO

CREATE TABLE [dbo].[TradeServiceUsers](
	[UserName] [varchar](50) NULL,
	[UserPWD] [varchar](max) NULL,
	[UserWatchlist] [varchar](max) NULL,
	[UserKey] [varchar](20) NULL,
	[Active] [bit] NULL,
	[UserSecret] [varchar](50) NULL,
	[Live_Paper] [bit] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO

