module.exports = {
	siteMetadata: {
		title: "Dylan's Blog",
		author: "Dylan Djian",
		description: "My personal blog on learning, implementing and discussing Machine Learning algorithms",
		siteUrl: "https://dylandjian.github.io"
	},
	pathPrefix: "/",
	plugins: [
		{
			resolve: `gatsby-source-filesystem`,
			options: {
				path: `${__dirname}/src/pages`,
				name: "pages"
			}
		},
		{
			resolve: `gatsby-transformer-remark`,
			options: {
				plugins: [
					{
						resolve: "gatsby-remark-embed-video",
						options: {
							width: 800,
							ratio: 1.77, // Optional: Defaults to 16/9 = 1.77
							height: 400, // Optional: Overrides optional.ratio
							related: false, //Optional: Will remove related videos from the end of an embedded YouTube video.
							noIframeBorder: true //Optional: Disable insertion of <style> border: 0
						}
					},
					{
						resolve: `gatsby-remark-images`,
						options: {
							maxWidth: 590
						}
					},
					{
						resolve: `gatsby-remark-responsive-iframe`,
						options: {
							wrapperStyle: `margin-bottom: 1.0725rem`
						}
					},
					{
						resolve: "gatsby-remark-prismjs",
						options: {
							classPrefix: "language-",
							inlineCodeMarker: null,
							aliases: {}
						}
					},
					"gatsby-remark-copy-linked-files",
					"gatsby-remark-embed-video",
					"gatsby-remark-responsive-iframe",
					"gatsby-remark-smartypants",
					`gatsby-remark-katex`
				]
			}
		},
		`gatsby-transformer-sharp`,
		`gatsby-plugin-sharp`,
		{
			resolve: `gatsby-plugin-google-analytics`,
			options: {
				//trackingId: `ADD YOUR TRACKING ID HERE`,
			}
		},
		`gatsby-plugin-feed`,
		`gatsby-plugin-offline`,
		`gatsby-plugin-react-helmet`,
		{
			resolve: "gatsby-plugin-typography",
			options: {
				pathToConfigModule: "src/utils/typography"
			}
		}
	]
}
