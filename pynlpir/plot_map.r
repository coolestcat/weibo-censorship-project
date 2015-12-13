library(maps)
library(mapdata)
library("maptools")
library(RColorBrewer)

x=readShapePoly('bou2_4p.shp')
x@data$NAME <- iconv(x@data$NAME, from = 'CP936', to = 'UTF-8')

mypalette_density<-brewer.pal(4,"YlOrRd")

densityColorCode <- data.frame(Density_category = c(0:4), 
                        Density_color = c("white", mypalette_density),
                        stringsAsFactors = FALSE)

censorship_ratios = read.csv("provinceTable.csv", header = TRUE)

density_color_df <- merge(censorship_ratios, 
                  densityColorCode, 
                  by = "Density_category")

getColor=function(mapdata,provname,provcol,othercol)
{
	f=function(x,y) ifelse(x %in% y,which(y==x),0)
	colIndex=sapply(mapdata@data$NAME,f,provname)
	fg=c(othercol,provcol)[colIndex+1]
	return(fg)
}

png('chinaplot.png')
plot(x, 
     col=getColor(x, density_color_df$Province, density_color_df$Density_color, "white"), 
     xlab="",ylab="")

title("Weibo Censorship Percentage by Province, China, 2012")
leg.txt <- c("<0.2%","0.2-0.3%", "0.3-0.4%", "0.4-0.5%", ">0.5%")
legend("bottom", leg.txt, horiz = TRUE, fill = densityColorCode$Density_color)
dev.off()

